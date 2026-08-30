MATERIAL_PRIORS = {

    "metal": {
        "density": (2700, 8900),
        "young": (5e10, 2.5e11),
        "poisson": (0.25, 0.35),
        "hardness": (1e9, 1e10),
        "friction": (0.2, 0.8),
    },

    "plastic": {
        "density": (850, 1500),
        "young": (5e8, 8e9),
        "poisson": (0.30, 0.45),
        "hardness": (5e7, 5e8),
        "friction": (0.2, 0.6),
    },

    "fabric": {
        "density": (300, 1500),
        "young": (1e6, 1e9),
        "poisson": (0.30, 0.49),
        "hardness": (1e5, 1e7),
        "friction": (0.4, 1.2),
    },

    "leather": {
        "density": (600, 1000),
        "young": (5e7, 5e8),
        "poisson": (0.35, 0.49),
        "hardness": (1e6, 5e7),
        "friction": (0.5, 1.0),
    },

    "foam": {
        "density": (20, 300),
        "young": (1e5, 5e8),
        "poisson": (0.20, 0.45),
        "hardness": (1e4, 1e6),
        "friction": (0.4, 1.0),
    },

    "rubber": {
        "density": (900, 1400),
        "young": (1e6, 1e7),
        "poisson": (0.45, 0.49),
        "hardness": (5e5, 5e7),
        "friction": (0.6, 1.5),
    },

    "glass": {
        "density": (2200, 2600),
        "young": (5e10, 9e10),
        "poisson": (0.20, 0.25),
        "hardness": (5e9, 1e10),
        "friction": (0.3, 0.9),
    },

    "stone": {
        "density": (2200, 3000),
        "young": (2e10, 7e10),
        "poisson": (0.15, 0.30),
        "hardness": (3e9, 1e10),
        "friction": (0.6, 1.2),
    },

    "ceramics": {
        "density": (2000, 6000),
        "young": (1e11, 4e11),
        "poisson": (0.20, 0.30),
        "hardness": (1e10, 3e10),
        "friction": (0.4, 1.0),
    },

    "wood": {
        "density": (300, 900),
        "young": (5e8, 2e10),
        "poisson": (0.25, 0.40),
        "hardness": (5e7, 5e8),
        "friction": (0.4, 0.9),
    },

    "concrete": {
        "density": (1800, 2600),
        "young": (2e10, 4e10),
        "poisson": (0.15, 0.25),
        "hardness": (1e9, 5e9),
        "friction": (0.6, 1.0),
    },

    # "other": {
    #     "density": (1, 2000),
    #     "young": (1e3, 1e9),
    #     "poisson": (0.0, 0.49),
    #     "hardness": (1e2, 1e7),
    #     "friction": (0.1, 1.5),
    # },
}
