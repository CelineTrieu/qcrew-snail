import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *

# Create SNAIL
n = 3
Lj = 11e-9
freq = 5.0e9


first_anharmonicity_list = []
fock_cutoff_list = []
average_anharmonicity_list = []
validity_list = []
a3_list = []
a4_list = []

alpha_list = np.arange(0.1, 0.6, 0.01)
phi_ext_list = np.arange(0.1 * 2 * np.pi, 0.5 * 2 * np.pi, 0.01 * 2 * np.pi)
for alpha in alpha_list:
    for phi_ext in phi_ext_list:
<<<<<<< HEAD
        snail = SNAIL(n, alpha, phi_ext, Ej=Ej)
        ancilla = Ancilla(snail, cap)
=======
        snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
        ancilla = Ancilla(snail, freq)
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828

        (
            first_anharmonicity,
            fock_cutoff,
            average_anharmonicity,
            is_average_reliable,
            a3,
            a4,
        ) = ancilla.analyze_anharmonicities()

        first_anharmonicity_list.append(first_anharmonicity)
        fock_cutoff_list.append(fock_cutoff)
        average_anharmonicity_list.append(average_anharmonicity)
        a3_list.append(a3)
        a4_list.append(a4)
        validity = 1 if is_average_reliable and not snail.has_multiple_wells else 0
        validity = validity * (
            1
            if -100 > first_anharmonicity > -400
            and average_anharmonicity < 30
            and fock_cutoff < 13
            else 0
        )
        validity_list.append(validity)

        print(
            "%.3f %.3f %.3f %.3f %.3f %d"
            % (
                first_anharmonicity,
                fock_cutoff,
                average_anharmonicity,
                alpha,
                phi_ext / 2 / np.pi,
                validity,
            )
        )

phi_ext_list /= 2 * np.pi  # plot in units of 2pi
reshape_dim = (len(alpha_list), len(phi_ext_list))

# Reshape values for colormesh plotting
first_anharmonicity_list = np.reshape(first_anharmonicity_list, reshape_dim)
fock_cutoff_list = np.reshape(fock_cutoff_list, reshape_dim)
average_anharmonicity_list = np.reshape(average_anharmonicity_list, reshape_dim)
validity_list = np.reshape(validity_list, reshape_dim)
a3_list = np.reshape(a3_list, reshape_dim)
a4_list = np.reshape(a4_list, reshape_dim)

fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)

im1 = axes[0, 0].pcolormesh(
    phi_ext_list, alpha_list, first_anharmonicity_list, shading="auto", cmap="bwr"
)
im1.set_clim(-500, +500)
fig.colorbar(im1, ax=axes[0, 0])
axes[0, 0].set_title("First anharmonicity")
axes[0, 0].set_ylabel("Josephson energy proportion alpha")

im2 = axes[1, 0].pcolormesh(
    phi_ext_list,
    alpha_list,
    average_anharmonicity_list,
    shading="auto",
    cmap="viridis_r",
)
im2.set_clim(0, 100)
fig.colorbar(im2, ax=axes[1, 0])
axes[1, 0].set_title("Avg high-N anharmonicity")
axes[1, 0].set_xlabel("External flux (per flux quanta)")
axes[1, 0].set_ylabel("Josephson energy proportion alpha")

im3 = axes[0, 1].pcolormesh(
    phi_ext_list, alpha_list, fock_cutoff_list, shading="auto", cmap="viridis_r"
)
im3.set_clim(0, 20)
fig.colorbar(im3, ax=axes[0, 1])
axes[0, 1].set_title("N cutoff")

im4 = axes[1, 1].pcolormesh(phi_ext_list, alpha_list, validity_list, shading="auto")
fig.colorbar(im4, ax=axes[1, 1])
axes[1, 1].set_title("Is a good set of parameters?")
axes[1, 1].set_xlabel("External flux (per flux quanta)")

im5 = axes[0, 2].pcolormesh(
    phi_ext_list, alpha_list, a3_list, shading="auto", cmap="viridis_r"
)
fig.colorbar(im5, ax=axes[0, 2])
axes[0, 2].set_title("a3 (RWA)")

im6 = axes[1, 2].pcolormesh(
    phi_ext_list, alpha_list, a4_list, shading="auto", cmap="bwr"
)
fig.colorbar(im6, ax=axes[1, 2])
axes[1, 2].set_title("a4 (RWA)")
axes[1, 2].set_xlabel("External flux (per flux quanta)")
# fig.colorbar(c, ax=ax)
plt.show()
