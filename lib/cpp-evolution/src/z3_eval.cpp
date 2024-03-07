
#include "architecture.h"
#include "experiment.h"
#include "mapping.h"

#if USE_Z3

#include <unordered_map>
#include <sstream>

#include"z3++.h"

using namespace z3;

#define X(i, k) (x_vars[(k) * num_uops + (i) ])

double Mapping::simulateExperiment(const Architecture &arch, const Experiment &e) const {
    std::unordered_map<Uop, num_type> uop_map;

    for (const auto& i : e.getInsnSeq()) {
        const auto &used_uops = *UopMap_.at(i);
        for (auto &[u, n] : used_uops) {
            uop_map[u] += n;
        }
    }

    int num_uops = uop_map.size();
    int num_ports = arch.getNumPorts();

    context c;
    optimize opt(c);

    expr t_var = c.real_const("t");

    std::vector<expr> x_vars;
    for (int k = 0; k < num_ports; ++k) {
        for (int i = 0; i < num_uops; ++i) {
            std::stringstream xname;
            xname << "x_" << i << "_" << k;
            x_vars.push_back(c.real_const(xname.str().c_str()));
        }
    }

    for (auto& x_var : x_vars) {
        opt.add(x_var >= c.real_val(0));
    }

    for (int k = 0; k < num_ports; ++k) {
        expr lhs = c.real_val(0);
        int i = 0;
        for (const auto &[u, n] : uop_map) {
            if ((u & (1 << k)) != 0) {
                lhs = lhs + X(i, k);
            }
            i++;
        }
        opt.add(lhs <= t_var);
    }

    int i = 0;
    for (const auto &[u, n] : uop_map) {
        expr lhs = c.real_val(0);
        for (int k = 0; k < num_ports; ++k) {
            if ((u & (1 << k)) != 0) {
                lhs = lhs + X(i, k);
            }
        }
        opt.add(lhs == c.real_val(n));

        i++;
    }

    // std::cout << opt << "\n";

    optimize::handle h = opt.minimize(t_var);
    if (sat == opt.check()) {
        expr res = opt.upper(h);
        return (double)res.numerator().get_numeral_int64() / (double)res.denominator().get_numeral_int64();
    }
    return -1.0;
}

#endif
