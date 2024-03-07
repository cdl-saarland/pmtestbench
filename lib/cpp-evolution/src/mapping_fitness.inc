#define FP_EPSILON 0.0001
#define INT_EPSILON 0
#define COMP_DISABLED (-1)

// This file contains the definition of the components of the fitness function.
// A component is defined by a line of the following format:
//   COMP(name, type, accepted error, group)
// Here, name is the name of the component, type its type (usually some float
// or integer type), accepted error is a constant of the right type that
// describes the maximal difference under which two values are still considered
// equal (particularly helpful for floats), and group is an integer constant
// greater than or equal to 0 or COMP_DISABLED.
// The group parameter indicates to which group of the fitness function the
// component belongs. The order of the component lines controls the
// lexicographic ordering within the groups.
//
// This file is used in the following structure:
//
// #define COMP(name, type, error, group) \
//     SOMETHING
// #include "mapping_fitness.inc"
// #undef COMP
//
// Where SOMETHING describes the action that is to be performed for each entry.

COMP(avg_err, float, FP_EPSILON, COMP_DISABLED)
COMP(avg_ipc_err, float, FP_EPSILON, 0)
COMP(singleton_avg_err, float, FP_EPSILON, COMP_DISABLED)
COMP(max_err, float, FP_EPSILON, COMP_DISABLED)
COMP(uop_volume, long int, INT_EPSILON, 1)
COMP(uop_number, long int, INT_EPSILON, COMP_DISABLED)
COMP(avg_num_diff_uops, float, FP_EPSILON, COMP_DISABLED)

#undef COMP_DISABLED
#undef FP_EPSILON

