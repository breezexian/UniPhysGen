CATEGORIES = {
    "Furniture": {
        "description": "Load-bearing or support objects for humans or items",
        "subcategories": [
            "SeatingFurniture",  # chair, sofa, stool
            "TableFurniture",  # desk, table, workbench
            "StorageFurniture",  # cabinet, shelf, wardrobe
            "BedAndSleepingFurniture",  # bed, bunk bed
            "OfficeFurniture",  # office desk, office chair
            "OutdoorFurniture",  # bench, patio table
            "FurnitureComponents",  # legs, handles, wheels
            "FurnitureAccessory",
            "Furniture_Other",  # extensible
        ],
    },

    "ArchitecturalBuilding": {
        "description": "Fixed building-level or structural objects",
        "subcategories": [
            "BuildingStructure",  # wall, ceiling, column
            "DoorWindowElement",  # door, window
            "ArchitecturalFixture",  # railing, stair
            "FlooringCovering",  # floor, carpet
            "BuildingComponent",  # beam, panel
            "Architecture_Other",
        ],
    },

    "LightingElectrical": {
        "description": "Lighting and electrical related objects",
        "subcategories": [
            "IndoorLighting",
            "OutdoorLighting",
            "LightingComponent",  # bulb, lamp head
            "ElectricalFixture",  # switch, socket
            "LightingElectrical_Other",
        ],
    },

    "PlumbingSanitary": {
        "description": "Water supply, drainage and sanitation",
        "subcategories": [
            "PlumbingFixture",  # toilet, sink
            "BathroomAccessory",  # towel rack
            "SanitationEquipment",
            "WaterControlComponent",  # valve, pipe
            "PlumbingSanitary_Other",
        ],
    },

    "KitchenDining": {
        "description": "Kitchen and dining related objects",
        "subcategories": [
            "KitchenAppliance",  # microwave, oven
            "CookwareUtensil",  # pot, knife
            "TablewareDrinkware",  # plate, cup
            "KitchenStorage",  # container, rack
            "KitchenDining_Other",
        ],
    },

    "ContainersStorage": {
        "description": "Non-furniture containers and vessels",
        "subcategories": [
            "GeneralContainer",  # box, bin
            "WasteRecyclingContainer",
            "PackagingBox",
            "StorageVessel",  # bottle, jar
            "Container_Other",
        ],
    },

    "ElectronicsDevices": {
        "description": "Electronic and digital devices",
        "subcategories": [
            "ConsumerElectronics",  # TV, camera
            "ComputerPeripheral",  # keyboard, mouse
            "AudioEquipment",
            "CommunicationDevice",  # phone, router
            "ControlInputDevice",  # controller
            "Electronics_Other",
        ],
    },

    "ToolsEquipment": {
        "description": "Tools and professional equipment",
        "subcategories": [
            "HandTool",
            "PowerTool",
            "IndustrialEquipment",
            "MeasurementInstrument",
            "ScientificInstrument",
            "ToolsEquipment_Other",
        ],
    },

    "SportsFitnessRecreation": {
        "description": "Sports, fitness and recreational objects",
        "subcategories": [
            "SportsEquipment",
            "FitnessEquipment",
            "RecreationalEquipment",
            "GameAccessory",
            "SportsFitness_Other",
        ],
    },

    "ToysFiguresModels": {
        "description": "Toys, figurines and display models",
        "subcategories": [
            "ToyObject",
            "GameObject",
            "FigurineCharacter",
            "EducationalModel",
            "ToyModel_Other",
        ],
    },

    "VehiclesMobility": {
        "description": "Vehicles and mobility-related objects",
        "subcategories": [
            "Vehicle",
            "VehicleComponent",
            "PersonalMobilityDevice",  # scooter, wheelchair
            "TransportAccessory",
            "Vehicle_Other",
        ],
    },

    "ApparelWearables": {
        "description": "Clothing, wearable and personal items",
        "subcategories": [
            "ApparelFootwear",
            "WearableDevice",
            "PersonalAccessory",
            "LuggageBag",
            "ApparelWearable_Other",
        ],
    },

    "PlantsGardenOutdoor": {
        "description": "Plants and outdoor-related objects",
        "subcategories": [
            "Plant",
            "Planter",
            "GardeningTool",
            "OutdoorFixture",  # pergola, fence
            "GardenOutdoor_Other",
        ],
    },

    "PublicInfrastructure": {
        "description": "Public space and infrastructure objects",
        "subcategories": [
            "PublicFurniture",  # bus stop bench
            "UrbanInfrastructure",  # traffic light
            "SafetySecurityEquipment",
            "PublicFacilityComponent",
            "PublicInfrastructure_Other",
        ],
    },
    "FoodBeverage": {
        "description": "Food, ingredients, snacks and drinks",
        "subcategories": [
            "RawIngredient",  # 肉类、蔬菜、水果、谷物等
            "CookedFood",  # 烹饪后的菜肴
            "Snack",  # 零食、糖果、面包等
            "Beverage",  # 水、饮料、酒等
            "FoodBeverage_Other",  # 其他无法归类的食物或饮料
        ],
    },
    "OtherMiscellaneous": {
        "description": "Objects that do not fit in the above 14 categories",
        "subcategories": [
            "OtherObject",
            "CustomObject",
            "CompositeObject",
        ],
    },
}
