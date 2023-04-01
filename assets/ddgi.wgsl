fn ls_probe_to_uv(N: vec3<f32>, ls_pos: vec3<f32>, color: bool) -> vec2<f32> {
    let ls_pos = clamp(ls_pos, vec3(0.0), CASF - 1.0);

    let probe_data_height = CAS.z * SIZE;

    let probe_tex_x = ls_pos.x * FSIZE + ls_pos.y * FSIZE * CASF.x;
    let probe_tex_y = ls_pos.z * FSIZE;

    let probe_uv = oct_encode_n(N, FSIZE, 1.0);

    let hit_probe_coord = vec2(probe_uv.x * FSIZE + probe_tex_x,
                               probe_uv.y * FSIZE + probe_tex_y);

    let tex_size = vec2<f32>(textureDimensions(ddgi_texture).xy);
    // TODO using probe_data_height * 2 for the blurred version was showing yellow 
    let ofs = select(0.0, f32(probe_data_height * 1), color);
    let uv = vec2<f32>(hit_probe_coord + vec2(0.0, ofs)) / tex_size;

    return uv;
}

fn sample_irradiance(N: vec3<f32>, ls_pos: vec3<f32>) -> vec3<f32> {
    let uv = ls_probe_to_uv(N, ls_pos, true);
    return textureSampleLevel(ddgi_texture, ddgi_texture_sampler, uv, 0.0).rgb;
}

fn sample_distance(N: vec3<f32>, ls_pos: vec3<f32>) -> vec3<f32> {
    let uv = ls_probe_to_uv(N, ls_pos, false);
    return textureSampleLevel(ddgi_texture, ddgi_texture_sampler, uv, 0.0).rgb;
}

fn irradiance_DDGI(N: vec3<f32>, V: vec3<f32>, ws_pos: vec3<f32>) -> vec3<f32> {
    //let NdotV = max(dot(N, V), 0.0001);
    //var fresnel = clamp(1.0 - NdotV, 0.0, 1.0);
    //fresnel = pow(fresnel, 2.0) * 0.5;
    //let refl_n = normalize(N * (2.0 * NdotV) - V);



    let probe_spacing = 1.5;
    var ws_offset = vec3(probe_spacing * -4.0, 0.5, probe_spacing * -4.0);
    ws_offset -= probe_spacing * 0.5;
    
    let cas_pos = ws_offset;

    let ls_pos = (ws_pos - cas_pos) / probe_spacing; 
    let ls_base_probe_pos = floor(ls_pos); //baseGridCoord
    let alpha = abs(ls_pos - ls_base_probe_pos);

    var sumIrradiance = vec3(0.0);
    var sumWeight = 0.0;
    
    // Iterate over adjacent probe cage
    for (var i = 0u; i < 8u; i += 1u) {

        // Compute the offset grid coord and clamp to the probe grid boundary
        // Offset = 0 or 1 along each axis
        let offset = vec3<u32>(i, i >> 1u, i >> 2u) & vec3<u32>(1u);
        let f_offset = vec3<f32>(offset);

        var ls_probe_coord = ls_base_probe_pos + f_offset; //probeGridCoord   
        //var p_idx = ls_to_idx(layer, ls_probe_coord);

        // Probe offset distance from grid
        // If probe ray hits backface, it will move probe to random position 
        // 0.4 from original grid
        
        // TODO uninplemented
        let ws_probe_offset = vec3(0.0);//probe_offsets(layer, p_idx);
        let backface = 0.0; //ws_probe_offset.w;

        // Make cosine falloff in tangent plane with respect to the angle from the surface to the probe so that we never
        // test a probe that is *behind* the surface.
        // It doesn't have to be cosine, but that is efficient to compute and we must clip to the tangent plane.
        let ws_probe_pos = ls_pos * probe_spacing + cas_pos;
                        //ls_to_ws(cas_pos, ls_probe_coord) + ws_probe_offset.xyz;

        // Bias the position at which visibility is computed; this
        // avoids performing a shadow test *at* a surface, which is a
        // dangerous location because that is exactly the line between
        // shadowed and unshadowed. If the normal bias is too small,
        // there will be light and dark leaks. If it is too large,
        // then samples can pass through thin occluders to the other
        // side (this can only happen if there are MULTIPLE occluders
        // near each other, a wall surface won't pass through itself.)
        let normalBias = 0.75; //TUNE default is 0.25
        let ws_probe_to_pos = ws_pos - ws_probe_pos + (N + 3.0 * V) * normalBias; //probeToPoint
        let ws_pos_to_probe_dir = normalize(-ws_probe_to_pos); //dir

        // Compute the trilinear weights based on the grid cell vertex to smoothly
        // transition between probes. Avoid ever going entirely to zero because that
        // will cause problems at the border probes. This isn't really a lerp. 
        // We're using 1-a when offset = 0 and a when offset = 1.
        let trilinear = mix(1.0 - alpha, alpha, f_offset);

        // Clamp all of the multiplies. We can't let the weight go to zero because then it would be 
        // possible for *all* weights to be equally low and get normalized
        // up to 1/n. We want to distinguish between weights that are 
        // low because of different factors.

        var weight = 1.0;

        // Probes that hit backfaces get 0, while front/sky hits are 1. Avoid using probes with lower 
        weight *= pow(backface, 12.0);

        // Smooth backface test
        {
            // Computed without the biasing applied to the "dir" variable. 
            // This test can cause reflection-map looking errors in the image
            // (stuff looks shiny) if the transition is poor.
            let trueDirectionToProbe = normalize(ws_probe_pos - ws_pos);

            // The naive soft backface weight would ignore a probe when
            // it is behind the surface. That's good for walls. But for small details inside of a
            // room, the normals on the details might rule out all of the probes that have mutual
            // visibility to the point. So, we instead use a "wrap shading" test below inspired by
            // NPR work.
            //weight *= max(0.0001, dot(trueDirectionToProbe, N));

            // The small offset at the end reduces the "going to zero" impact
            // where this is really close to exactly opposite
            weight *= square(max(0.0001, (dot(trueDirectionToProbe, N) + 1.0) * 0.5)) + 0.2; 
        }

        // Moment visibility test
        {
            let dist_to_probe = length(ws_probe_to_pos);

            var temp = sample_distance(-ws_pos_to_probe_dir, ls_pos);
            let mean = temp.x;
            let variance = abs(square(temp.x) - temp.y);

            // http://www.punkuser.net/vsm/vsm_paper.pdf; equation 5
            // Need the max in the denominator because biasing can cause a negative displacement
            var chebyshevWeight = variance / (variance + square(max(dist_to_probe - mean, 0.0))); 

            // Increase contrast in the weight 
            // TUNE is usually max(cube(chebyshevWeight), 0.0)
            chebyshevWeight = max(cube(chebyshevWeight), 0.0);

            if dist_to_probe > mean {
                weight *= chebyshevWeight;
            }

        }

        // Sample probe with both normal and reflection normal
        //var probe_col_data = irradianceSH_2n(layer, N, refl_n, p_idx);
        var ir_col = sample_irradiance(N, ls_probe_coord);

        //var ir_col = probe_col_data[0];
        //let refl_col = probe_col_data[1];

        //ir_col += fresnel * refl_col;



        weight = pow(weight, 0.15); // Griffin TUNE blurs more between probes 0.25

        // Avoid zero weight
        weight = max(0.000001, weight);

        var probeIrradiance = max(ir_col.rgb, vec3(0.0));

        // A tiny bit of light is really visible due to log perception, so
        // crush tiny weights but keep the curve continuous. This must be done
        // before the trilinear weights, because those should be preserved.
        let crushThreshold = 0.2; //TUNE default is 0.2
        if (weight < crushThreshold) {
            weight *= weight * weight * (1.0 / square(crushThreshold)); 
        }

        // Trilinear weights
        weight *= trilinear.x * trilinear.y * trilinear.z;

        // Weight in a more-perceptual brightness space instead of radiance space.
        // This softens the transitions between probes with respect to translation.
        // It makes little difference most of the time, but when there are radical transitions
        // between probes this helps soften the ramp.
        probeIrradiance = sqrt(probeIrradiance); //OPTIONAL (but does seem better)

        sumIrradiance += weight * probeIrradiance;
        sumWeight += weight;
    }
    
    var netIrradiance = max(sumIrradiance / sumWeight, vec3(0.0));

    // Go back to linear irradiance
    netIrradiance = square_vec3(netIrradiance); //OPTIONAL (but does seem better)

    netIrradiance *= 0.85; //TUNE energyPreservation

    let lambertianIndirect = 0.5 * PI * netIrradiance;

    return lambertianIndirect;
}