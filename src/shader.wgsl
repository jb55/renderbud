const PI: f32 = 3.14159265;
const EPS: f32 = 1e-4;

const MIN_ROUGHNESS: f32 = 0.04;
const AMBIENT_INTENSITY: f32 = 0.6;
const INV_GAMMA: f32 = 1.0 / 2.2;

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
@group(2) @binding(3) var ao_mr_tex: texture_2d<f32>;
@group(2) @binding(4) var normal_tex: texture_2d<f32>;

struct VSIn {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv:  vec2<f32>,
  @location(3) tangent: vec4<f32>,
};

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) world_tangent: vec4<f32>, // xyz + w
};


@vertex
fn vs_main(v: VSIn) -> VSOut {
  var out: VSOut;

  // For now: identity model matrix. Next step is per-object transforms.
  let world4 = object.model * vec4<f32>(v.pos, 1.0);
  out.world_pos = world4.xyz;

  let n4 = object.normal * vec4<f32>(v.normal, 0.0);
  let t4 = object.model * vec4<f32>(v.tangent.xyz, 0.0);

  out.world_normal = normalize(n4.xyz);
  out.uv = v.uv;
  out.clip = globals.view_proj * world4;
  out.world_tangent = vec4<f32>(normalize(t4.xyz), v.tangent.w);
  return out;
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn saturate3(v: vec3<f32>) -> vec3<f32> { return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)); }

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= EPS) { return vec3<f32>(0.0, 0.0, 1.0); }
  return v * inverseSqrt(l2);
}

// Fresnel (Schlick)
fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  let ct = saturate(cosTheta);
  let f = pow(1.0 - ct, 5.0);
  return F0 + (1.0 - F0) * f;
}

// GGX / Trowbridge-Reitz NDF
fn ggx_ndf(NdotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith geometry with Schlick-GGX (UE4 k)
fn smith_g_schlick_ggx(NdotV: f32, NdotL: f32, alpha: f32) -> f32 {
  let k = alpha + 1.0;
  let k2 = (k * k) / 8.0;

  let gv = NdotV / (NdotV * (1.0 - k2) + k2);
  let gl = NdotL / (NdotL * (1.0 - k2) + k2);
  return gv * gl;
}

fn diffuse_lambert(diffuseColor: vec3<f32>) -> vec3<f32> {
  return diffuseColor / PI;
}

// RG normal map decode (optionally flips Y for your convention)
fn decode_normal_rg(tex: vec3<f32>) -> vec3<f32> {
  let x = tex.r * 2.0 - 1.0;
  var y = tex.g * 2.0 - 1.0;
  y = -y; // <- keep your current behavior here

  let z2 = max(1.0 - x*x - y*y, 0.0);
  let z = sqrt(z2);
  return safe_normalize(vec3<f32>(x, y, z));
}

fn build_tbn(Ng: vec3<f32>, world_tangent: vec4<f32>) -> mat3x3<f32> {
  var T = safe_normalize(world_tangent.xyz);
  T = safe_normalize(T - Ng * dot(Ng, T));
  let B = safe_normalize(cross(Ng, T)) * world_tangent.w;
  return mat3x3<f32>(T, B, Ng);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let Ng = safe_normalize(in.world_normal);
  let tbn = build_tbn(Ng, in.world_tangent);

  let n_ts = decode_normal_rg(textureSample(normal_tex, material_sampler, in.uv).xyz);
  let N = safe_normalize(tbn * n_ts);

  let V = safe_normalize(globals.cam_pos - in.world_pos);
  let L = safe_normalize(-globals.light_dir);
  let H = safe_normalize(V + L);

  let NdotL = saturate(dot(N, L));
  let NdotV = saturate(dot(N, V));
  let NdotH = saturate(dot(N, H));
  let VdotH = saturate(dot(V, H));

  let bc_tex = textureSample(basecolor_tex, material_sampler, in.uv);
  let ao_mr  = textureSample(ao_mr_tex, material_sampler, in.uv);

  // glTF metallicRoughnessTexture: G=roughness, B=metallic
  let baseColor = bc_tex.rgb * material.base_color_factor.rgb;
  let metallic  = saturate(ao_mr.b * material.metallic_factor);
  let rough_in  = ao_mr.g * material.roughness_factor;

  // AO: R channel; strength lerp from 1 -> ao
  let ao_tex = ao_mr.r;
  let ao = 1.0 + (ao_tex - 1.0) * saturate(material.ao_strength);
  //let ao = 1.0;

  let roughness = clamp(rough_in, MIN_ROUGHNESS, 1.0);
  let alpha = roughness * roughness;

  let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let diffuseColor = baseColor * (1.0 - metallic);

  let D = ggx_ndf(NdotH, alpha);
  let Gs = smith_g_schlick_ggx(NdotV, NdotL, alpha);
  let F = fresnel_schlick(VdotH, F0);

  let denom = max(4.0 * NdotV * NdotL, EPS);
  let spec = (D * Gs) * F / denom;

  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let diff = kd * diffuse_lambert(diffuseColor);

  let direct = (diff + spec) * (globals.light_color * NdotL);
  let ambient = diffuseColor * AMBIENT_INTENSITY * ao;

  var col = direct + ambient;

  // simple tonemap + gamma
  col = col / (col + vec3<f32>(1.0));

  // we have to gamma correct in an egui context
  col = pow(col, vec3<f32>(INV_GAMMA));

  return vec4<f32>(saturate3(col), 1.0);
}
