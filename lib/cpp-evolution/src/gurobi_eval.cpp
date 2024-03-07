
#include "architecture.h"
#include "experiment.h"
#include "mapping.h"

#include <unordered_map>

#define WRITE_MODELS 0
// #define WRITE_MODELS 1

#if USE_GUROBI

#include "gurobi_c++.h"

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

    static GRBEnv env;
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_Method, 2); // grbtune said that this is a good idea
#if GUROBI_SINGLETHREAD
    env.set(GRB_IntParam_Threads, 1);
#endif

    GRBModel model(env);

    GRBVar t_var = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "t");

    GRBVar* x_vars = model.addVars(nullptr, nullptr, nullptr, nullptr, nullptr, num_uops * num_ports);

    for (int k = 0; k < num_ports; ++k) {
        GRBLinExpr lhs = 0;
        int i = 0;
        for (const auto &[u, n] : uop_map) {
            if ((u & (1 << k)) != 0) {
                lhs += X(i, k);
            }
            i++;
        }
        model.addConstr(lhs, GRB_LESS_EQUAL, t_var);
    }

    int i = 0;
    for (const auto &[u, n] : uop_map) {
        GRBLinExpr lhs = 0;
        for (int k = 0; k < num_ports; ++k) {
            if ((u & (1 << k)) != 0) {
                lhs += X(i, k);
            }
        }
        model.addConstr(lhs, GRB_EQUAL, n);

        i++;
    }

    GRBLinExpr obj = t_var;
    model.setObjective(obj, GRB_MINIMIZE);

#if WRITE_MODELS
    model.write("model.mps");
#endif
    model.optimize();

    if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
        delete[] x_vars;
        return -1.0;
    }

    double res = model.get(GRB_DoubleAttr_ObjVal);
    delete[] x_vars;

    return res;
}

#endif
