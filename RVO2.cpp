#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <RVO.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Constants
const int num_agents = 2;            // Number of agents
const int ensemble_size = 2000;      // Ensemble size
const int state_size = 4;            // State dimensions: [x, y, vx, vy]
const int max_iterations = 32;       // Maximum simulation steps
const double noise_std = 0.05;        // Standard deviation for Gaussian noise

// Global variables
RVO::RVOSimulator* sim = nullptr;

// Define the state structure for each agent
struct AgentState {
    double x_pos;
    double y_pos;
    double v_x;
    double v_y;
};

// Define a structure to hold target positions
struct TargetPosition {
    double x;
    double y;
};

// Function to read CSV and load data into agent1_data and agent2_data vectors
void load_csv(const std::string& file_path, std::vector<AgentState>& agent1_data, std::vector<AgentState>& agent2_data) {
    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return;
    }

    // Skip the header
    std::getline(file, line);

    // Read each line of the CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        AgentState agent1_state;
        AgentState agent2_state;

        // Read the time value (ignore if not needed)
        std::getline(ss, value, ',');  // Skip the 'time' column

        // Read data for agent 1
        std::getline(ss, value, ','); agent1_state.x_pos = std::stod(value);  // x_pos_1
        std::getline(ss, value, ','); agent1_state.y_pos = std::stod(value);  // y_pos_1
        std::getline(ss, value, ','); agent1_state.v_x = std::stod(value);    // v_x1
        std::getline(ss, value, ','); agent1_state.v_y = std::stod(value);    // v_y1

        // Read data for agent 2
        std::getline(ss, value, ','); agent2_state.x_pos = std::stod(value);  // x_pos_2
        std::getline(ss, value, ','); agent2_state.y_pos = std::stod(value);  // y_pos_2
        std::getline(ss, value, ','); agent2_state.v_x = std::stod(value);    // v_x2
        std::getline(ss, value);       agent2_state.v_y = std::stod(value);   // v_y2

        // Store the state for both agents
        agent1_data.push_back(agent1_state);
        agent2_data.push_back(agent2_state);
    }

    file.close();
}

// Function to calculate entropy based on the diagonal covariance vector Q_diag
double calculate_entropy(const std::vector<double>& Q_diag) {
    size_t state_sz = Q_diag.size();
    double det_Q = 1.0;

    // Compute determinant (product of diagonal elements)
    for (size_t i = 0; i < state_sz; ++i) {
        double diag_element = Q_diag[i];
        if (std::isnan(diag_element) || diag_element <= 0.0) {
            // Assign a small positive value to avoid log of zero or negative
            diag_element = 1e-6;
        }
        det_Q *= diag_element;
    }

    // Ensure determinant is positive
    if (det_Q <= 0.0 || std::isnan(det_Q)) {
        det_Q = 1e-24;
    }

    // Calculate entropy using the formula:
    // e(Q_tilda) = 1/2 * (log((2πe)^d * det(Q_tilda)))
    double entropy = 0.5 * (std::log(std::pow(2.0 * M_PI * M_E, state_sz)) + std::log(det_Q));

    // Check for NaN entropy
    if (std::isnan(entropy)) {
        entropy = 0.0;
    }

    return entropy;
}


// Add global ramp-up counters for each agent
const size_t ramp_up_steps = 20; // Number of steps to ramp up velocity
std::vector<size_t> ramp_up_counter(num_agents, 0);

// Modify the update_agent_pref_velocity function
void update_agent_pref_velocity(RVO::RVOSimulator* sim, size_t agent_id, const TargetPosition& target) {
    RVO::Vector2 current_pos = sim->getAgentPosition(agent_id);

    // Calculate the direction vector towards the target
    RVO::Vector2 target_pos(static_cast<float>(target.x), static_cast<float>(target.y));
    RVO::Vector2 direction_vector = target_pos - current_pos;
    float distance = RVO::abs(direction_vector); // Euclidean distance

    // Define a threshold to consider the target as reached
    const float threshold = 0.1f; // Adjust as necessary

    if (distance < threshold) {
        std::cout << "Agent " << agent_id << " has reached the target. Stopping movement." << std::endl;
        sim->setAgentPrefVelocity(agent_id, RVO::Vector2(0.0f, 0.0f));
    }
    else {
        // Normalize the direction
        RVO::Vector2 direction = RVO::normalize(direction_vector);

        // Implement damping: reduce speed as the agent approaches the target
        float max_speed = sim->getAgentMaxSpeed(agent_id);
        float desired_speed = max_speed * (distance / 5.0f); // Adjust divisor based on desired damping

        // Clamp the speed to the maximum allowed speed
        if (desired_speed > max_speed) {
            desired_speed = max_speed;
        }

        // Implement ramp-up: gradually increase preferred velocity from zero to desired_speed
        /*if (ramp_up_counter[agent_id] < ramp_up_steps) {
            float ramp_factor = static_cast<float>(ramp_up_counter[agent_id]) / ramp_up_steps;
            desired_speed *= ramp_factor;
            ramp_up_counter[agent_id]++;
        }*/

        // Smoothly interpolate preferred velocity to prevent abrupt changes
        RVO::Vector2 current_pref_vel = sim->getAgentPrefVelocity(agent_id);
        RVO::Vector2 new_pref_vel = current_pref_vel + (direction * desired_speed - current_pref_vel) * 0.3f; // 0.5f is the interpolation factor

        // Set the updated preferred velocity
        sim->setAgentPrefVelocity(agent_id, new_pref_vel);
    }
}

// Function to setup the RVO simulator with parameters
void setupRVOSimulator(RVO::RVOSimulator* sim, const std::vector<AgentState>& initial_states) {
	sim->setTimeStep(0.588f); // Time step of the simulation

    // Set agent defaults: neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
    sim->setAgentDefaults(5.0f, 2, 1.5f, 5.0f, 0.1f, 0.68f);

    // Add agents to the simulator based on initial positions from real-world data
    for (const auto& agent_state : initial_states) {
        sim->addAgent(RVO::Vector2(static_cast<float>(agent_state.x_pos), static_cast<float>(agent_state.y_pos)));
    }
}

// Function to update agent positions and velocities using RVO
void updateRVOSimulator(RVO::RVOSimulator* sim, std::vector<AgentState>& current_states) {
    // Perform a simulation step
    sim->doStep();

    // Update the current_states with new positions and velocities
    for (size_t i = 0; i < num_agents; ++i) {
        RVO::Vector2 pos = sim->getAgentPosition(i);
        RVO::Vector2 vel = sim->getAgentVelocity(i);
        current_states[i].x_pos = pos.x();
        current_states[i].y_pos = pos.y();
        current_states[i].v_x = vel.x();
        current_states[i].v_y = vel.y();
    }
}

// Separate history vectors for each agent
std::vector<std::vector<double>> Q_cov_history_agent1;
std::vector<std::vector<double>> Q_cov_history_agent2;

// Kalman Gain calculation and Correction Step
void kalman_gain_and_correction(
    const AgentState& agent1_measurement,
    const AgentState& agent2_measurement,
    std::vector<std::vector<AgentState>>& ensemble,
    std::vector<std::vector<double>>& P1,      // State covariance matrix for Agent 1
    std::vector<double>& Q_cov1,               // Process noise covariance vector for Agent 1
    std::vector<std::vector<double>>& P2,      // State covariance matrix for Agent 2
    std::vector<double>& Q_cov2,               // Process noise covariance vector for Agent 2
    const std::vector<std::vector<double>>& R_cov  // Measurement noise covariance matrix
) {
    size_t state_sz = 4;  // [x_pos, y_pos, v_x, v_y]

    // Predict Step for Agent 1: P1 = P1 + Q_cov1 (only diagonal elements)
    for (size_t i = 0; i < state_sz; ++i) {
        P1[i][i] += Q_cov1[i];
    }

    // Predict Step for Agent 2: P2 = P2 + Q_cov2 (only diagonal elements)
    for (size_t i = 0; i < state_sz; ++i) {
        P2[i][i] += Q_cov2[i];
    }

    // Calculate Kalman Gain for Agent 1: K1 = P1 / (P1 + R_cov)
    std::vector<double> K1(state_sz);
    for (size_t i = 0; i < state_sz; ++i) {
        K1[i] = P1[i][i] / (P1[i][i] + R_cov[i][i]);
    }

    // Calculate Kalman Gain for Agent 2: K2 = P2 / (P2 + R_cov)
    std::vector<double> K2(state_sz);
    for (size_t i = 0; i < state_sz; ++i) {
        K2[i] = P2[i][i] / (P2[i][i] + R_cov[i][i]);
    }

    // Update ensemble members
    for (size_t n = 0; n < ensemble_size; ++n) {
        // Agent 1 correction
        ensemble[n][0].x_pos += K1[0] * (agent1_measurement.x_pos - ensemble[n][0].x_pos);
        ensemble[n][0].y_pos += K1[1] * (agent1_measurement.y_pos - ensemble[n][0].y_pos);
        ensemble[n][0].v_x += K1[2] * (agent1_measurement.v_x - ensemble[n][0].v_x);
        ensemble[n][0].v_y += K1[3] * (agent1_measurement.v_y - ensemble[n][0].v_y);

        // Agent 2 correction
        ensemble[n][1].x_pos += K2[0] * (agent2_measurement.x_pos - ensemble[n][1].x_pos);
        ensemble[n][1].y_pos += K2[1] * (agent2_measurement.y_pos - ensemble[n][1].y_pos);
        ensemble[n][1].v_x += K2[2] * (agent2_measurement.v_x - ensemble[n][1].v_x);
        ensemble[n][1].v_y += K2[3] * (agent2_measurement.v_y - ensemble[n][1].v_y);
    }

    // Update State Covariance P1: P1 = (I - K1) * P1 + Regularization
    for (size_t i = 0; i < state_sz; ++i) {
        P1[i][i] = (1.0 - K1[i]) * P1[i][i] + 1e-6;  // Regularization added
    }

    // Update State Covariance P2: P2 = (I - K2) * P2 + Regularization
    for (size_t i = 0; i < state_sz; ++i) {
        P2[i][i] = (1.0 - K2[i]) * P2[i][i] + 1e-6;  // Regularization added
    }
}

// Function to perform Maximum Likelihood Estimation (MLE) for updating Q_cov and apply smoothing
void mle_per_agent(
    const AgentState& agent1_measurement,
    const AgentState& agent2_measurement,
    const std::vector<std::vector<AgentState>>& ensemble,
    std::vector<double>& Q_cov1, // Changed to vector<double>
    std::vector<double>& Q_cov2, // Changed to vector<double>
    size_t timestep  // Pass the current timestep
) {
    size_t w = 2;  // Window size on each side (total window size = 2w +1 =5)
    size_t window_size = w;
    size_t total_window = 2 * w + 1;

    size_t state_sz = 4;  // [x_pos, y_pos, v_x, v_y]

    // Initialize covariance sums for Agent 1 and Agent 2
    std::vector<double> sum_Q1(state_sz, 0.0);
    std::vector<double> sum_Q2(state_sz, 0.0);

    // Compute sum of squared differences
    for (size_t n = 0; n < ensemble_size; ++n) {
        // Differences for Agent 1
        double diff1_x = agent1_measurement.x_pos - ensemble[n][0].x_pos;
        double diff1_y = agent1_measurement.y_pos - ensemble[n][0].y_pos;
        double diff1_vx = agent1_measurement.v_x - ensemble[n][0].v_x;
        double diff1_vy = agent1_measurement.v_y - ensemble[n][0].v_y;

        // Differences for Agent 2
        double diff2_x = agent2_measurement.x_pos - ensemble[n][1].x_pos;
        double diff2_y = agent2_measurement.y_pos - ensemble[n][1].y_pos;
        double diff2_vx = agent2_measurement.v_x - ensemble[n][1].v_x;
        double diff2_vy = agent2_measurement.v_y - ensemble[n][1].v_y;

        // Sum squared differences for Agent 1
        sum_Q1[0] += diff1_x * diff1_x;
        sum_Q1[1] += diff1_y * diff1_y;
        sum_Q1[2] += diff1_vx * diff1_vx;
        sum_Q1[3] += diff1_vy * diff1_vy;

        // Sum squared differences for Agent 2
        sum_Q2[0] += diff2_x * diff2_x;
        sum_Q2[1] += diff2_y * diff2_y;
        sum_Q2[2] += diff2_vx * diff2_vx;
        sum_Q2[3] += diff2_vy * diff2_vy;
    }

    // Update Q_cov1 with averaged values and regularization
    for (size_t i = 0; i < state_sz; ++i) {
        Q_cov1[i] += (sum_Q1[i] / ensemble_size) + 1e-6;

        // Ensure Q_cov1[i] is positive
        if (Q_cov1[i] < 1e-6 || std::isnan(Q_cov1[i])) {
            Q_cov1[i] = 1e-6;
        }

        // Update Q_cov2 with averaged values and regularization
        Q_cov2[i] = (sum_Q2[i] / ensemble_size) + 1e-6;

        // Ensure Q_cov2[i] is positive
        if (Q_cov2[i] < 1e-6 || std::isnan(Q_cov2[i])) {
            Q_cov2[i] = 1e-6;
        }
    }

    // Add the updated Q_cov to history for both agents
    Q_cov_history_agent1.push_back(Q_cov1);
    Q_cov_history_agent2.push_back(Q_cov2);

    // Perform smoothing if enough history is available
    if (Q_cov_history_agent1.size() >= total_window && Q_cov_history_agent2.size() >= total_window) {
        // Initialize smoothed Q_cov for Agent 1 and Agent 2
        std::vector<double> smoothed_Q_cov1(state_sz, 0.0);
        std::vector<double> smoothed_Q_cov2(state_sz, 0.0);

        // Compute the smoothed Q_cov for Agent 1
        for (size_t w_idx = Q_cov_history_agent1.size() - total_window; w_idx < Q_cov_history_agent1.size(); ++w_idx) {
            for (size_t i = 0; i < state_sz; ++i) {
                smoothed_Q_cov1[i] += Q_cov_history_agent1[w_idx][i];
            }
        }
        for (size_t i = 0; i < state_sz; ++i) {
            smoothed_Q_cov1[i] /= static_cast<double>(total_window);
        }

        // Compute the smoothed Q_cov for Agent 2
        for (size_t w_idx = Q_cov_history_agent2.size() - total_window; w_idx < Q_cov_history_agent2.size(); ++w_idx) {
            for (size_t i = 0; i < state_sz; ++i) {
                smoothed_Q_cov2[i] += Q_cov_history_agent2[w_idx][i];
            }
        }
        for (size_t i = 0; i < state_sz; ++i) {
            smoothed_Q_cov2[i] /= static_cast<double>(total_window);
        }

        // Assign the smoothed Q_cov vectors
        Q_cov1 = smoothed_Q_cov1;
        Q_cov2 = smoothed_Q_cov2;
    }

    // Limit the size of Q_cov_history to total_window
    while (Q_cov_history_agent1.size() > total_window) {
        Q_cov_history_agent1.erase(Q_cov_history_agent1.begin());
    }
    while (Q_cov_history_agent2.size() > total_window) {
        Q_cov_history_agent2.erase(Q_cov_history_agent2.begin());
    }
}

int main() {
    // Data vectors for real-world positions and velocities for agent 1 and agent 2
    std::vector<AgentState> agent1_data;
    std::vector<AgentState> agent2_data;

    // Load the real-world data from the CSV file for both agents
    load_csv("F:\\Thesis\\LARGE-SCALE-CROWD-SIMULATION\\simulation_agent_positions_and_velocities_complete.csv", agent1_data, agent2_data);

    // Check if data was loaded
    if (agent1_data.empty() || agent2_data.empty()) {
        std::cerr << "Error: Agent data is empty. Please check the CSV file." << std::endl;
        return -1;
    }

    // Initialize ensemble states for both agents
    std::vector<std::vector<AgentState>> ensemble(ensemble_size, std::vector<AgentState>(num_agents));

    // Initialize covariance matrices: Q_cov1, P1 for Agent 1; Q_cov2, P2 for Agent 2
    std::vector<std::vector<double>> P1(state_size, std::vector<double>(state_size, 0.0));
    std::vector<double> Q_cov1(state_size, 0.0);
    std::vector<std::vector<double>> P2(state_size, std::vector<double>(state_size, 0.0));
    std::vector<double> Q_cov2(state_size, 0.0);
    std::vector<std::vector<double>> R_cov(state_size, std::vector<double>(state_size, 0.0));

    // Set diagonal elements appropriately for Agent 1 and Agent 2
    for (size_t i = 0; i < state_size; ++i) {
        P1[i][i] = 0.1;
        Q_cov1[i] = 0.05; // Increased from 0.1 to 0.16 to ensure positive entropy
        P2[i][i] = 0.1;
        Q_cov2[i] = 0.05; // Increased from 0.1 to 0.16 to ensure positive entropy
        R_cov[i][i] = 0.5; // Reduced for higher measurement trust
    }

    // Initialize Q_cov_history with the initial Q_cov for both agents
    Q_cov_history_agent1.push_back(Q_cov1);
    Q_cov_history_agent2.push_back(Q_cov2);

    // Define target positions for each agent
    std::vector<TargetPosition> targets(num_agents);
    // Assign targets appropriately
    // Example assignments 
    //targets[0].x = 15.25; // Replace with actual target
    //targets[0].y = 1.7;
    //targets[1].x = 5.17; // Replace with actual target
    //targets[1].y = 2.02;

    targets[0].x = agent1_data.back().x_pos; // Agent 0's target X
    targets[0].y = agent1_data.back().y_pos; // Agent 0's target Y
    targets[1].x = agent2_data.back().x_pos; // Agent 1's target X
    targets[1].y = agent2_data.back().y_pos; // Agent 1's target Y

    // Initialize RVO simulator
    sim = new RVO::RVOSimulator();

    // Prepare initial states
    std::vector<AgentState> initial_states(num_agents);
    initial_states[0] = agent1_data[0];
    initial_states[1] = agent2_data[0];

    // Setup RVO simulator with initial agent positions
    setupRVOSimulator(sim, initial_states);

    // Open a file to save the simulation results
    std::ofstream results_file("simulation_results.csv");
    if (!results_file.is_open()) {
        std::cerr << "Error: Could not open simulation_results.csv for writing." << std::endl;
        return -1;
    }
    results_file << "time,agent1_x,agent1_y,agent2_x,agent2_y,entropy\n";

	std::cout << targets[0].x << " " << targets[0].y << " " << targets[1].x << " " << targets[1].y << std::endl;

    // Main simulation loop for each timestep
    for (size_t t = 0; t < max_iterations; ++t) {
        std::cout << "---------- Timestep " << t << " ----------" << std::endl;
        
        // Step 1: Update each agent's preferred velocity towards its own target
        for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
            update_agent_pref_velocity(sim, agent_id, targets[agent_id]);
        }

        // Step 2: Perform a single simulation step
        updateRVOSimulator(sim, initial_states);

        // Step 3: Generate ensemble predictions by adding Gaussian noise to current states
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> noise_dist(0.0, noise_std);

        /*for (size_t i = 0; i < ensemble_size; ++i) {
            for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
                ensemble[i][agent_id].x_pos = initial_states[agent_id].x_pos + noise_dist(gen);
                ensemble[i][agent_id].y_pos = initial_states[agent_id].y_pos + noise_dist(gen);
                ensemble[i][agent_id].v_x = initial_states[agent_id].v_x + noise_dist(gen);
                ensemble[i][agent_id].v_y = initial_states[agent_id].v_y + noise_dist(gen);
            }
        }*/
        // Implement state-dependent noise,compare to above where velocity components might receive less noise compared to position components, ensuring stability in velocity damping
        std::normal_distribution<double> pos_noise_dist(0.0, noise_std);
        std::normal_distribution<double> vel_noise_dist(0.0, noise_std / 2);

        for (size_t i = 0; i < ensemble_size; ++i) {
            for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
                ensemble[i][agent_id].x_pos = initial_states[agent_id].x_pos + pos_noise_dist(gen);
                ensemble[i][agent_id].y_pos = initial_states[agent_id].y_pos + pos_noise_dist(gen);
                ensemble[i][agent_id].v_x = initial_states[agent_id].v_x + vel_noise_dist(gen);
                ensemble[i][agent_id].v_y = initial_states[agent_id].v_y + vel_noise_dist(gen);
            }
        }

        // Step 4: Apply Kalman Gain and correction for both agents
        AgentState agent1_measurement;
        AgentState agent2_measurement;

        if (t < agent1_data.size() && t < agent2_data.size()) {
            agent1_measurement = agent1_data[t];
            agent2_measurement = agent2_data[t];
        }
        else {
            agent1_measurement = agent1_data.back();
            agent2_measurement = agent2_data.back();
        }

        kalman_gain_and_correction(agent1_measurement, agent2_measurement, ensemble, P1, Q_cov1, P2, Q_cov2, R_cov);

        // Step 5: Update covariance matrix Q using MLE and apply smoothing
        mle_per_agent(agent1_measurement, agent2_measurement, ensemble, Q_cov1, Q_cov2, t);

        // Step 6: Calculate entropy based on updated and smoothed covariance Q for both agents
        double entropy1 = calculate_entropy(Q_cov1);
        double entropy2 = calculate_entropy(Q_cov2);
        double total_entropy = (entropy1 + entropy2)/2;
        std::cout << "Entropy at timestep " << t << ": " << total_entropy << std::endl;

        // Step 7: Save the positions and entropy to the results file
        results_file << t << ","
            << initial_states[0].x_pos << ","
            << initial_states[0].y_pos << ","
            << initial_states[1].x_pos << ","
            << initial_states[1].y_pos << ","
            << total_entropy << "\n";

        // Print the positions and velocities of the agents
        for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
            std::cout << "Agent " << agent_id + 1 << ": Position("
                << initial_states[agent_id].x_pos << ", "
                << initial_states[agent_id].y_pos << "), Velocity("
                << initial_states[agent_id].v_x << ", "
                << initial_states[agent_id].v_y << ")" << std::endl;
        }

        // save the position velocities of the agents in a csv file as Agent1_x, Agent1_y, Agent1_vx, Agent1_vy, Agent2_x, Agent2_y, Agent2_vx, Agent2_vy
        /*std::ofstream agent_file("C:\\Users\\harsh\\OneDrive - IIT Kanpur\\Desktop\\Thesis\\simulated_positions_and_velocitiy_new.csv", std::ios::app);
		agent_file << initial_states[0].x_pos << "," << initial_states[0].y_pos << "," << initial_states[0].v_x << "," << initial_states[0].v_y << "," << initial_states[1].x_pos << "," << initial_states[1].y_pos << "," << initial_states[1].v_x << "," << initial_states[1].v_y << "," << total_entropy << std::endl;   
        agent_file.close();*/

        // Check if all agents have reached their targets
        bool all_reached = true;
        for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
            double dx = initial_states[agent_id].x_pos - targets[agent_id].x;
            double dy = initial_states[agent_id].y_pos - targets[agent_id].y;
            double distance = std::sqrt(dx * dx + dy * dy);
            if (distance > 0.1) { // Threshold distance
                all_reached = false;
                break;
            }
        }
        if (all_reached) {
            std::cout << "All agents have reached their targets." << std::endl;
            break;
        }

        // Step 8: Update the initial states for the next iteration
        for (size_t agent_id = 0; agent_id < num_agents; ++agent_id) {
            double mean_x = 0.0, mean_y = 0.0, mean_vx = 0.0, mean_vy = 0.0;
            for (size_t n = 0; n < ensemble_size; ++n) {
                mean_x += ensemble[n][agent_id].x_pos;
                mean_y += ensemble[n][agent_id].y_pos;
                mean_vx += ensemble[n][agent_id].v_x;
                mean_vy += ensemble[n][agent_id].v_y;
            }
            mean_x /= ensemble_size;
            mean_y /= ensemble_size;
            mean_vx /= ensemble_size;
            mean_vy /= ensemble_size;

            initial_states[agent_id].x_pos = mean_x;
            initial_states[agent_id].y_pos = mean_y;
            initial_states[agent_id].v_x = mean_vx;
            initial_states[agent_id].v_y = mean_vy;
        }
    }

    // Close the results file
    results_file.close();
    // Clean up the simulator
    delete sim;
    return 0;
}