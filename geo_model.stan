data{

    int n_timesteps;
    int treatment_time_idx;
    int n_regions;
    int n_treated;
    array[n_treated] int treated_region_id;
    array[n_regions - n_treated] int control_region_id;
    matrix[n_timesteps, n_regions] Y;

}

transformed data {

    // Need to scale the columns of y
    int n_treated_regions = n_treated;
    int n_control_regions = n_regions - n_treated_regions;

    int n_timesteps_pre = treatment_time_idx;
    int n_timesteps_post = n_timesteps - n_timesteps_pre;

    matrix[n_timesteps_pre, n_control_regions] y_control_pre = Y[1:n_timesteps_pre, control_region_id];
    matrix[n_timesteps_post, n_control_regions] y_control_post = Y[(n_timesteps_pre+1):n_timesteps, control_region_id ];

    matrix[n_timesteps_pre, n_treated_regions] y_treatment_pre = Y[1:n_timesteps_pre, treated_region_id];
    matrix[n_timesteps_post, n_treated_regions] y_treatment_post = Y[(n_timesteps_pre+1):n_timesteps, treated_region_id ];

    matrix[n_timesteps_pre, n_control_regions] scaled_y_control_pre;
    matrix[n_timesteps_post, n_control_regions] scaled_y_control_post;

    matrix[n_timesteps_pre, n_treated_regions] scaled_y_treatment_pre;
    matrix[n_timesteps_post, n_treated_regions] scaled_y_treatment_post;

    vector[n_treated_regions] treatment_column_means;
    vector[n_treated_regions] treatment_column_sds;

    vector[n_control_regions] control_column_means;
    vector[n_control_regions] control_column_sds;

    for(i in 1:n_control_regions){
        control_column_means[i] = mean(y_control_pre[:, i]);
        control_column_sds[i] = sd(y_control_pre[:, i]);

        scaled_y_control_pre[:, i] = (y_control_pre[:, i] - control_column_means[i]) / control_column_sds[i];
        scaled_y_control_post[:, i] = (y_control_post[:, i] - control_column_means[i]) / control_column_sds[i];
    }

    for(i in 1:n_treated_regions){
        treatment_column_means[i] = mean(y_treatment_pre[:, i]);
        treatment_column_sds[i] = sd(y_treatment_pre[:, i]);

        scaled_y_treatment_pre[:, i] = (y_treatment_pre[:, i] - treatment_column_means[i]) / treatment_column_sds[i];
        scaled_y_treatment_post[:, i] = (y_treatment_post[:, i] - treatment_column_means[i]) / treatment_column_sds[i];
    }

    // Hyperparameter for prior on the weights
    vector[n_control_regions] alpha = rep_vector(1.0, n_control_regions);   

}

parameters {
    //array[n_treated_regions] simplex[n_control_regions] region_weights;
    array[n_treated_regions] vector[n_control_regions] region_weights;
    array[n_treated_regions] real<lower=0> sigma;
}

model {
    // implements the sum-to-1-constraint
    for(i in 1:n_treated_regions) {
        //target += dirichlet_lpdf(region_weights[i] | alpha);
        target += normal_lpdf(region_weights[i] | 0, 1);;
        target += cauchy_lpdf(sigma[i] | 0, 1);
        target += normal_id_glm_lpdf(scaled_y_treatment_pre[:, i] | scaled_y_control_pre, 0.0, region_weights[i], sigma[i]);
    }

}

generated quantities {
    matrix[n_timesteps_pre, n_treated_regions] insample_fit;
    matrix[n_timesteps_post, n_treated_regions] expected_counterfactual;
    matrix[n_timesteps_post, n_treated_regions] predicted_counterfactual;
    matrix[n_timesteps_post, n_treated_regions] treatment_effect_on_treated;
    matrix[n_timesteps_post, n_treated_regions] cumulative_treatment_effects;
    matrix[n_timesteps_post, n_treated_regions] estimated_lift;
    vector[n_treated_regions] lift;
    real total_lift;

    for(j in 1:n_treated_regions){
        insample_fit[:, j] = scaled_y_control_pre * region_weights[j].* treatment_column_sds[j] + treatment_column_means[j];
        expected_counterfactual[:, j] = (scaled_y_control_post * region_weights[j]) .* treatment_column_sds[j] + treatment_column_means[j];
        predicted_counterfactual[:, j] = to_vector(normal_rng(expected_counterfactual[:, j], sigma[j]));
        treatment_effect_on_treated[:, j] = y_treatment_post[:, j] - predicted_counterfactual[:, j];
        cumulative_treatment_effects[:, j] = cumulative_sum(treatment_effect_on_treated[:, j]);
        estimated_lift[:, j] =  treatment_effect_on_treated[:, j] ./ predicted_counterfactual[:, j];
        lift[j] = sum(treatment_effect_on_treated[:, j]) / sum(predicted_counterfactual[:, j]); 
    }

    total_lift = sum(treatment_effect_on_treated) / sum(predicted_counterfactual);
}
