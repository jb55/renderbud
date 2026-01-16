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

@group(0) @binding(0) var<uniform> G: Globals;

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

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn ggx_ndf(NdotH: f32, a: f32) -> f32 {
  let a2 = a*a;
  let d = (NdotH*NdotH)*(a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}

fn smith_g(NdotV: f32, NdotL: f32, a: f32) -> f32 {
  // Schlick-GGX geometry
  let k = (a + 1.0);
  let k2 = (k*k) / 8.0;
  let gv = NdotV / (NdotV*(1.0 - k2) + k2);
  let gl = NdotL / (NdotL*(1.0 - k2) + k2);
  return gv * gl;
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
  // screen ray
  let aspect = G.resolution.x / G.resolution.y;
  let p = (in.uv * 2.0 - vec2<f32>(1.0)) * vec2<f32>(aspect, 1.0);

  let ro = G.cam_pos;
  let rd = normalize(vec3<f32>(p, -1.3));

  // scene: one sphere + ground plane feel (via background)
  let sphere_c = vec3<f32>(0.0, 0.0, 0.0);
  let t = ray_sphere(ro, rd, sphere_c, 0.6);

  // background gradient (helps sell “game lighting”)
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

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    // choose a stylized base color (like a hero/AC item)
    let baseColor = vec3<f32>(0.25, 0.60, 0.95);

    let rim = pow(1.0 - NdotV, 2.0);
    let rimCol = rim * vec3<f32>(0.20, 0.25, 0.35);

    // --- ACNH-ish: soft diffuse + broad spec + warm ambient ---
    let ambient = vec3<f32>(0.22, 0.24, 0.26);
    let wrap = 0.35; // diffuse wrap for softness
    let ndl_wrap = saturate((dot(N, L) + wrap) / (1.0 + wrap));

    // gentle, broad highlight
    let specPow = 96.0;
    let spec = pow(saturate(dot(reflect(-L, N), V)), specPow) * 0.15;

    col = baseColor * (ambient + ndl_wrap * G.light_color * 0.9) + spec * G.light_color + rimCol * 0.6;

    // subtle “toy-like” tonemap
    col = col / (col + vec3<f32>(0.7));
    col = pow(col, vec3<f32>(1.0/1.8)); // a touch of lift
  }

  return vec4<f32>(sat3(col), 1.0);
}
