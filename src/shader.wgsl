const PI: f32 = 3.14159265;

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

    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct Object {
  model: mat4x4<f32>,
  normal: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> object: Object;

struct Material {
  base_color_factor: vec4<f32>,
  metallic_factor: f32,
  roughness_factor: f32,
  ao_strength: f32,
  _pad0: f32,
};

@group(2) @binding(0) var<uniform> material: Material;
@group(2) @binding(1) var material_sampler: sampler;
@group(2) @binding(2) var basecolor_tex: texture_2d<f32>;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var normal_tex: texture_2d<f32>;
@group(2) @binding(5) var ao_tex: texture_2d<f32>;

struct VSIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv:  vec2<f32>,
};

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VSIn) -> VSOut {
  var out: VSOut;

  // For now: identity model matrix. Next step is per-object transforms.
  let world4 = object.model * vec4<f32>(v.pos, 1.0);
  out.world_pos = world4.xyz;

  let n4 = object.normal * vec4<f32>(v.normal, 0.0);

  out.world_normal = normalize(n4.xyz);
  out.uv = v.uv;
  out.clip = globals.view_proj * world4;

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
  let G = globals;

  let N = normalize(in.world_normal);
  let V = normalize(G.cam_pos - in.world_pos);
  let L = normalize(-G.light_dir);
  let H = normalize(V + L);

  let NoL = saturate(dot(N, L));
  let NoV = saturate(dot(N, V));
  let NoH = saturate(dot(N, H));
  let VoH = saturate(dot(V, H));

  let m = material;
  let bc_tex = textureSample(basecolor_tex, material_sampler, in.uv);
  let mr = textureSample(mr_tex, material_sampler, in.uv);
  let ao_s = textureSample(ao_tex, material_sampler, in.uv);

  // glTF: metallicRoughnessTexture: G=roughness, B=metallic
  let baseColor = bc_tex.rgb * m.base_color_factor.rgb;
  let metallic: f32 = saturate(mr.b * m.metallic_factor);
  let roughness_in: f32 = mr.g * m.roughness_factor;

  // AO: R channel; strength as art-directable lerp from 1 -> ao
  let ao_tex_v: f32 = ao_s.r;
  let ao: f32 = 1.0 + (ao_tex_v - 1.0) * saturate(m.ao_strength);

  let roughness = clamp(roughness_in, 0.04, 1.0);
  let alpha = roughness * roughness;

  let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let diffuseColor = baseColor * (1.0 - metallic);

  let D = ggx_ndf(NoH, alpha);
  let Gg = smith_g_schlick_ggx(NoV, NoL, alpha);
  let F = fresnel_schlick(VoH, F0);

  let denom = max(4.0 * NoV * NoL, 1e-4);
  let spec = (D * Gg) * F / denom;

  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let diff = kd * diffuse_lambert(diffuseColor);

  let direct = (diff + spec) * (G.light_color * NoL);

  let ambientIntensity = 0.25;
  let ambient = diffuseColor * ambientIntensity * ao;

  var col = direct + ambient;

  let rim = pow(1.0 - NoV, 2.0) * 0.08;
  col += rim * vec3<f32>(0.8, 0.9, 1.0);

  col = col / (col + vec3<f32>(1.0));
  col = pow(col, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(sat3(col), 1.0);
}


