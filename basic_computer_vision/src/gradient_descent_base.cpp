#include "../include/gradient_descent_base.h"
#include <algorithm>

GradientDescentBase::GradientDescentBase(double step_size)
    : step_size_(step_size) {
}

void GradientDescentBase::run(int max_iteration) {
    std::cout.precision(5);
    initialize();
    last_energy_ = compute_energy();
    std::cout << "init energy : " << last_energy_ << "@@@@@@" << '\n';
    int current_iter = 0;
    while (!is_terminate(current_iter, max_iteration)) 
	{
        current_iter++;
        update();

        double new_energy = compute_energy();
        std::cout << return_drive_class_name() << " : "
                  << "||current iteration|| : " << current_iter << " ";
        std::cout << "||current step size|| : " << step_size_ << " ";
        if (new_energy < last_energy_) {
            std::cout << "  engery decresed, accept update , "
                      << " ||new energy|| : " << new_energy
                      << " ||last energy|| : " << last_energy_;
            std::cout << " energy decresed for: " << new_energy - last_energy_
                      << '\n';
            last_energy_ = new_energy;
        } else {
            std::cout << "  engery incresed, "
                      << " ||new energy|| : " << new_energy
                      << " ||last energy|| : " << last_energy_ << '\n';
            std::cout << " energy incresed for: " << new_energy - last_energy_
                      << '\n';
        }
    }
    print_terminate_info();
}

bool GradientDescentBase::is_terminate(int current_iter,
                                       int max_iteration) const {
    return (current_iter >= max_iteration);
}
void GradientDescentBase::print_terminate_info() const {
    std::cout << "Iteration finished" << std::endl;
}

void GradientDescentBase::update_step_size(bool is_energy_decent) {
    step_size_ = std::max(std::min(1.0, step_size_), 1e-60);
}