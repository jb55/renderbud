let PI: f32 = 3.14159265;

struct Globals {
    time: f32,
    _pad0: f32,
    resolution: vec2<f32>,

    cam_pos: vec3<f32>,
    _pad3: f32,

    light_dir: vec3<f32>,
    _pad1: f32,

    light_color: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  // Full-screen triangle
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  var out: VSOut;
  out.pos = vec4<f32>(p[vid], 0.0, 1.0);
  out.uv = (out.pos.xy * 0.5) + vec2<f32>(0.5);
  return out;
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)); }

// Fresnel (Schlick)
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// GGX / Trowbridge-Reitz NDF
fn ggx_ndf(NdotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith geometry with Schlick-GGX
fn smith_g_schlick_ggx(NdotV: f32, NdotL: f32, alpha: f32) -> f32 {
  // k from UE4 style
  let k = (alpha + 1.0);
  let k2 = (k * k) / 8.0;

  let gv = NdotV / (NdotV * (1.0 - k2) + k2);
  let gl = NdotL / (NdotL * (1.0 - k2) + k2);
  return gv * gl;
}

// Lambert diffuse (simple, stable)
fn diffuse_lambert(diffuseColor: vec3<f32>) -> vec3<f32> {
  return diffuseColor / PI;
}

fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let oc = ro - c;
  let b = dot(oc, rd);
  let c2 = dot(oc, oc) - r*r;
  let h = b*b - c2;
  if (h < 0.0) { return -1.0; }
  return -b - sqrt(h);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // alias
  let G = globals;

  // screen ray
  let aspect = G.resolution.x / G.resolution.y;
  let p = (in.uv * 2.0 - vec2<f32>(1.0)) * vec2<f32>(aspect, 1.0);

  let ro = G.cam_pos;
  let rd = normalize(vec3<f32>(p, -1.3));

  let sphere_c = vec3<f32>(0.0, 0.0, 0.0);
  let t = ray_sphere(ro, rd, sphere_c, 0.6);

  // background
  var col = mix(
      vec3<f32>(0.10, 0.12, 0.15),
      vec3<f32>(0.02, 0.03, 0.05),
      in.uv.y
  );

  if (t > 0.0) {
    let pos = ro + rd * t;
    let N = normalize(pos - sphere_c);
    let V = normalize(ro - pos);
    let L = normalize(-G.light_dir);
    let H = normalize(V + L);

    let NoL = saturate(dot(N, L));
    let NoV = saturate(dot(N, V));
    let NoH = saturate(dot(N, H));
    let VoH = saturate(dot(V, H));

    // --- "glTF material" constants for now (later: textures) ---
    // Base color in linear space (when using textures, decode sRGB -> linear)
    let baseColor = vec3<f32>(0.25, 0.60, 0.95);

    // Metallic-roughness workflow
    let metallic: f32 = 0.05;       // try 0..1
    let roughness_in: f32 = 0.65;   // try 0..1

    // AO / "flat darkness": 1 = none, 0 = very occluded
    let ao: f32 = 0.75;

    // Avoid singular highlights
    let roughness = clamp(roughness_in, 0.04, 1.0);
    let alpha = roughness * roughness;

    // glTF standard mixing:
    // Dielectric F0 ≈ 0.04, metals use baseColor as F0
    let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
    let diffuseColor = baseColor * (1.0 - metallic);

    // Specular BRDF
    let D = ggx_ndf(NoH, alpha);
    let Gg = smith_g_schlick_ggx(NoV, NoL, alpha);
    let F = fresnel_schlick(VoH, F0);

    let denom = max(4.0 * NoV * NoL, 1e-4);
    let spec = (D * Gg) * F / denom;

    // Energy conservation (diffuse reduced by Fresnel)
    let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diff = kd * diffuse_lambert(diffuseColor);

    // Direct lighting
    let direct = (diff + spec) * (G.light_color * NoL);

    // Ambient (fake IBL): AO only affects ambient/indirect
    // This is the "ACNH-ish darkness" knob.
    let ambientIntensity = 0.25;
    let ambient = diffuseColor * ambientIntensity * ao;

    col = direct + ambient;

    // Optional: tiny rim for readability (stylized, but subtle)
    let rim = pow(1.0 - NoV, 2.0) * 0.08;
    col += rim * vec3<f32>(0.8, 0.9, 1.0);

    // Simple tonemap + gamma
    col = col / (col + vec3<f32>(1.0));
    col = pow(col, vec3<f32>(1.0 / 2.2));
  }

  return vec4<f32>(sat3(col), 1.0);
}


