#pragma once

#include <stdint.h>
#include <vector>
#include <cmath>

using seed_type = uint32_t;

class Config {
public:
    int getPopulationSize(void) const {
        return PopulationSize;
    }

    int getMaxChildNum(void) const {
        return (int)ceil((double)getPopulationSize() * (MaxRecombinationFactor + MaxMutationFactor));
    }

    double getMaxRecombinationFactor(void) const {
        return MaxRecombinationFactor;
    }

    double getMaxMutationFactor(void) const {
        return MaxMutationFactor;
    }

    int getNumIterations(void) const {
        return NumIterations;
    }

    int getNumEpochs(void) const {
        return NumEpochs;
    }

    double getKeepRatio(void) const {
        return KeepRatio;
    }

    double getLuckChance(void) const {
        return LuckChance;
    }

    double getBadLuckProtection(void) const {
        return BadLuckProtection;
    }

    int getNumPorts(void) const {
        return NumPorts;
    }

    double getMutAddUopChance(void) const {
        return MutAddUopChance;
    }

    double getMutChangeUopChance(void) const {
        return MutChangeUopChance;
    }

    double getMutChangeNumChance(void) const {
        return MutChangeNumChance;
    }

    bool getEnableLocalOptimization(void) const {
        return EnableLocalOptimization;
    }

    bool getEnableRatioCombination(void) const {
        return EnableRatioCombination;
    }

    bool getEnableSymmetricError(void) const {
        return EnableSymmetricError;
    }

    double getPeakIPC(void) const {
        return PeakIPC;
    }

    friend class ConfigParser;

private:
    int PopulationSize = 200;
    double MaxRecombinationFactor = 1.0;
    double MaxMutationFactor = 1.0;
    int NumIterations = 100;
    int NumEpochs = 3;
    double KeepRatio = 0.1;

    double LuckChance = 0.1;
    double BadLuckProtection = 0.05;

    double MutAddUopChance = 0.04;
    double MutChangeUopChance = 0.04;
    double MutChangeNumChance = 0.04;

    bool EnableLocalOptimization = true;
    bool EnableRatioCombination = false;

    bool EnableSymmetricError = false;

    int NumPorts = 8;
    double PeakIPC = -1.0;
};

