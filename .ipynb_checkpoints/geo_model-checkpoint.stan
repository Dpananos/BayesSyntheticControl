data{

    int n_timesteps_pre;
    int n_timesteps_post;

    int n_control_regions;
    int n_treatment_regions;

    matrix[n_timesteps_pre, n_control_regions] y_control_pre;
    matrix[n_timesteps_post, n_control_regions] y_control_post;
    
    matrix[n_timesteps_pre, n_treatment_regions] y_treatment_pre;
    matrix[n_timesteps_post, n_treatment_regions] y_treatment_post;
    int sample;
    int sum_to_one_constraint;

}

transformed data {

    vector[n_control_regions] alpha = rep_vector(1.0, n_control_regions);   
   
    // Need to scale the columns of y

    vector[n_control_regions] control_column_means;
    vector[n_control_regions] control_column_sds;

    vector[n_control_regions] treatment_column_means;
    vector[n_control_regions] treatment_column_sds;

    matrix[n_timesteps_pre, n_control_regions] scaled_y_control_pre;
    matrix[n_timesteps_post, n_control_regions] scaled_y_control_post;
    
    matrix[n_timesteps_pre, n_treatment_regions] scaled_y_treatment_pre;
    matrix[n_timesteps_post, n_treatment_regions] scaled_y_treatment_post;

    real xbar;
    real sds;



    for(j in 1:n_control_regions) {

        control_column_means[j] = mean(y_control_pre[, j]);
        control_column_sds[j] = sd(y_control_pre[, j]);

        xbar = control_column_means[j];
        sds = control_column_sds[j];

        scaled_y_control_pre[:, j] = (y_control_pre[, j] - xbar) / sds;
        scaled_y_control_post[:, j] = (y_control_post[, j] - xbar) / sds;


    }

    for(j in 1:n_treatment_regions) {

        treatment_column_means[j] = mean(y_treatment_pre[, j]);
        treatment_column_sds[j] = sd(y_treatment_pre[, j]);

        xbar = treatment_column_means[j];
        sds = treatment_column_sds[j];

        scaled_y_treatment_pre[:, j] = (y_treatment_pre[, j] - xbar) / sds;
        scaled_y_treatment_post[:, j] = (y_treatment_post[, j] - xbar) / sds;
        


    }
}

parameters {
    array[n_treatment_regions] simplex[n_control_regions] region_weights;
    //array[n_treatment_regions] vector[n_control_regions] region_weights;

    array[n_treatment_regions] real<lower=0> sigma;
}

model {
    // implements the sum-to-1-constraint
    for(i in 1:n_treatment_regions) {

        if(sum_to_one_constraint){
            target += dirichlet_lpdf(region_weights[i] | alpha);
        }else{
            target += student_t_lpdf(region_weights[i] | 30.0, 0, 1);
        }

        target += cauchy_lpdf(sigma[i] | 0, 1);

        if(sample>0){
            target += normal_id_glm_lpdf(scaled_y_treatment_pre[:, i] | scaled_y_control_pre, 0.0, region_weights[i], sigma[i]);
        }
    }

}

generated quantities {
    matrix[n_timesteps_pre, n_treatment_regions] insample_fit;
    matrix[n_timesteps_post, n_treatment_regions] expected_counterfactual;
    matrix[n_timesteps_post, n_treatment_regions] predicted_counterfactual;
    matrix[n_timesteps_post, n_treatment_regions] treatment_effect_on_treated;
    matrix[n_timesteps_post, n_treatment_regions] cumulative_treatment_effects;
    matrix[n_timesteps_post, n_treatment_regions] estimated_lift;
    vector[n_treatment_regions] average_lift;
    real total_lift;
    real naive_average_lift;

    for(j in 1:n_treatment_regions){
        insample_fit[:, j] = scaled_y_control_pre * region_weights[j].* treatment_column_sds[j] + treatment_column_means[j];
        expected_counterfactual[:, j] = (scaled_y_control_post * region_weights[j]) .* treatment_column_sds[j] + treatment_column_means[j];
        predicted_counterfactual[:, j] = to_vector(normal_rng(expected_counterfactual[:, j], sigma[j]));
        treatment_effect_on_treated[:, j] = y_treatment_post[:, j] - predicted_counterfactual[:, j];
        cumulative_treatment_effects[:, j] = cumulative_sum(treatment_effect_on_treated[:, j]);
        estimated_lift[:, j] =  treatment_effect_on_treated[:, j] ./ predicted_counterfactual[:, j];
        average_lift[j] = sum(treatment_effect_on_treated[:, j]) / sum(predicted_counterfactual[:, j]); 
    }

    total_lift = sum(treatment_effect_on_treated) / sum(predicted_counterfactual);
    naive_average_lift = mean(average_lift);
}
