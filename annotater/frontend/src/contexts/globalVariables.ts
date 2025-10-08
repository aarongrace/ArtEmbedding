import { TaskType } from "../baseClasses/Ant";

export const defaultVars = {
    // Ant-related stats
    ant: {
        workerBaseSpeed: 0.06,
        soldierBaseSpeed: 0.04,
        queenBaseSpeed: 0.03,
        workerSpeedMult: 1.0,
        soldierSpeedMult: 1.0,
        workerAttack: 2,
        soldierAttack: 10,
        workerFoodConsumption: 0.17, // per half second
        soldierFoodConsumption: 0.4,
        queenFoodConsumption: 3,
        workerHpRecoveryRate: 0.4,
        soldierHpRecoveryRate: 0.8,
        queenHpRecoveryRate: 0.9,
        foodConsumptionScaleFactor: 0.13,
        workerCarryingCapacity: 5,
        soldierCarryingCapacity: 7,
        idleSpeedFactor: 0.5,
        patrolRange: 250,
        idleRange: 100,
    },

    // UI-related settings
    ui: {
        canvasWidth: 600, // Moved from SettingsStore
        canvasHeight: 400, // Moved from SettingsStore
        canvasProportions: {
            width: 0.7,
            height: 0.77,
        },
        carriedEntitySize: {
            width: 15,
            height: 15,
        },
        edgeMargin: 12,
        selectedCircleRadius: 20,
        scrollDirection: {x: 0, y: 0},
        scrollEdgeOffset: 50,
        scrollSpeed: 1.5, 
        scrollDelay: 100,
        remainingScrollDelay: 100,
    },

    food: {
        foodPerAnt: 20,
        foodWasteBaseline: 200,
        decayFactor: 1.3,
        fruitSize: 40,
        defaultFruitAmount: 50,
        entitySpawnFactor: 1.5,
        minDistanceBetweenEntities: 50,
        growFruitRateOfDecrease: 0.2,
        chitinCollectRate: 0.1,
    },

    // Reproduction-related settings
    reproduction: {
        eggChance: 0.06,
        maxEggs: 10,
    },

    update: {
        syncInterval: 3000,
        discreteUpdateInterval: 500,
    },

    // Enemy-related settings
    enemy: {
        baseEnemySpawnChance: 0.05,
    },

    highlightedTasks: [] as TaskType[],
    managingPatrol: false,
    seasonLength: 30,
    season: 0,
    offline_mode: false,
    dashBoardInitialized: false,
};

export let vars = structuredClone(defaultVars);
