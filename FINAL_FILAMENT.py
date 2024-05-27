import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk
from tqdm import tqdm
import os

# Definir argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Simulación de filamento en 3D con fuerzas y movimiento browniano.')
parser.add_argument('--radio', type=float, default=20.0, help='Radio de la esfera')
parser.add_argument('--fuerza', type=float, default=0.1, help='Valor de la fuerza activa')

args = parser.parse_args()

Numparticles = 50

"""
force parameters
"""
factive = args.fuerza  # Active force
K_spring = 10.0
epsilon_LJ = 1.0
sigma_LJ = 1.0
rcut_LJ = sigma_LJ * 2 ** (1 / 6)
l0 = 0.86 * rcut_LJ + 1e-3
"""
Integrator parameters
"""
dt = 0.005
kBT = 0.1
gamma = 1.0

"""
Sphere parameters
"""
sphere_center = np.array([0.0, 0.0, 0.0])  # Centro de la esfera
sphere_radius = args.radio  # Radio de la esfera

"""
Simulation parameters
"""
tsimul = 1000
print("###########################################################")
print(f'Radio de la esfera: {sphere_radius}')
print(f'Número de partículas: {Numparticles}')
print(f'Fuerza activa: {factive}')
print("###########################################################")
###############################################
outfolder = f'N={Numparticles}_fuerza{args.fuerza}_radio_{args.radio}'
os.makedirs(outfolder, exist_ok=True)

###############################################
def create_filament_particles_dict(num_particles):
    filament_dict = {}
    for i in range(1, num_particles - 1):  # Corrected loop range
        filament_dict[i] = {'From': i, 'Next': i + 1, 'Prev': i - 1}
    filament_dict[0] = {'From': 0, 'Next': 1, 'Prev': None}  # Initial particle adjustment
    filament_dict[num_particles - 1] = {'From': num_particles - 1, 'Next': None,
                                        'Prev': num_particles - 2}  # Last particle adjustment
    return filament_dict


def create_filament_positions_3D(num_particles, l0):
    # Posiciones iniciales de las partículas
    positions = np.zeros((num_particles, 3))
    for i in range(num_particles):
        positions[i] = [i * l0, 0, 0]  # Se asignan posiciones en el eje x
    return positions


def plot_filament_3D(_positions, r0, output_file='filament'):
    """
    Plot the structure of a filament based on given positions in 3D.

    Parameters:
    _positions (np.array): Numpy array of shape (n, 3), where each row contains x, y, and z coordinates.
    output_file (str): The base name of the output file where the plot will be saved.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_positions = _positions[:, 0]
    y_positions = _positions[:, 1]
    z_positions = _positions[:, 2]

    for x, y, z in zip(x_positions, y_positions, z_positions):
        ax.scatter(x, y, z, color='r')

    # Plot the spherical surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center[0]
    y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center[1]
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center[2]
    ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(output_file)
    plt.savefig(f'{output_file}.png', dpi=200)
    plt.close(fig)


def save_vtk(_positions, output_file):
    """
        Output the system as a VTP file for direct visualisation in ParaView.
        Parameter
        ---------
        outfile : string
            Name of the output file.
        Note
        ----
        Caller should ensure that the file has correct extension
    """
    points = vtk.vtkPoints()
    ids = vtk.vtkIntArray()
    pos = vtk.vtkDoubleArray()
    ids.SetNumberOfComponents(1)
    pos.SetNumberOfComponents(3)
    ids.SetName("id")
    pos.SetName("pos")
    id = 0
    for r in _positions:
        points.InsertNextPoint(r.tolist())
        ids.InsertNextValue(id)
        pos.InsertNextTuple(r.tolist())
        id += 1
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().AddArray(ids)
    polyData.GetPointData().AddArray(pos)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file + ".vtp")
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polyData)
    else:
        writer.SetInputData(polyData)
    writer.SetDataModeToAscii()
    writer.Write()


# Funciones de potencial y fuerza en 3D
def lennard_jones_force_fn(r):
    r6 = r ** 6
    return 24 * epsilon_LJ * (2 * (sigma_LJ ** 6) / r6 ** 2 - sigma_LJ ** 12 / r6) / r ** 2


hookean_force = lambda x: -K_spring * x


# Función para calcular fuerzas en 3D
def compute_forces_3D(positions, filament_dict, l0):
    forces = np.zeros_like(positions)
    for particle_i in range(0, positions.shape[0]):
        particle_info_i = filament_dict[particle_i]
        position_i = positions[particle_i]

        for particle_j in range(particle_i + 1, positions.shape[0]):
            particle_info_j = filament_dict[particle_j]
            position_j = positions[particle_j]
            rij = position_i - position_j
            rij_distance = np.linalg.norm(rij + 1e-6)  # Evitar divisi�n por cero
            if rij_distance < rcut_LJ:
                # Calculate Lennard-Jones interaction force
                force = lennard_jones_force_fn(rij_distance)
                forces[particle_i] += force * rij / rij_distance
                forces[particle_j] -= force * rij / rij_distance

        if particle_info_i['Prev'] is not None:
            particle_i_minus_1 = positions[particle_info_i['Prev']]
            x_i_prev = position_i - particle_i_minus_1
            x_i_distance = np.linalg.norm(x_i_prev)
            # Calculate Hookean force
            force = hookean_force(x_i_distance - l0)
            forces[particle_i] += force * x_i_prev / x_i_distance
            x_i_prev /= x_i_distance
            # Add active force (tangent vector)
            forces[particle_i] += factive * x_i_prev

        if particle_info_i['Next'] is not None:
            particle_i_plus_1 = positions[particle_info_i['Next']]
            x_i_next = position_i - particle_i_plus_1
            x_i_distance = np.linalg.norm(x_i_next)
            # Calculate Hookean force
            force = hookean_force(x_i_distance - l0)
            forces[particle_i] += force * x_i_next / x_i_distance
            x_i_next /= -x_i_distance
            # Add active force (tangent vector)
            forces[particle_i] += factive * x_i_next
    return forces


def brownian_motion_3D(_positions, _forces, _dt, _gamma, _kBT):
    brownian_force_displacement = np.random.normal(0, _gamma * _kBT * np.array([1.0, 1.0, 1.0]),
                                                   (_positions.shape)) * np.sqrt(dt)
    _positions += (_gamma * _forces * _dt + brownian_force_displacement)
    return _positions


def project_on_sphere(positions, center, radius):
    v = positions - center
    r = np.linalg.norm(v, axis=1)
    projected_positions = center + v * (radius / r)[:, np.newaxis]
    return projected_positions


# Crear el filamento
filament_dict = create_filament_particles_dict(Numparticles)
positions = create_filament_positions_3D(Numparticles, l0)

# Plot inicial del filamento
plot_filament_3D(positions, r0=rcut_LJ, output_file=f'{outfolder}/filament_initial')

"""
Aqui tienes que equilibrar el sistema antes de la simulacion
por al menos 500 pasos de tiempo poniendo la fuerza activa en 0
"""
factive_old = factive
factive = 0.0
tsimul_equilibration = 100
for t in (pbar := tqdm(range(tsimul_equilibration))):
    pbar.set_description(f"Equilibrando {int}")
    # Calcular fuerzas
    for i in range(0, int(1 / dt)):
        forces = compute_forces_3D(positions, filament_dict, l0)
        # Realizar movimiento browniano
        positions = brownian_motion_3D(positions, forces, dt, gamma, kBT)
        # Proyectar posiciones sobre la superficie esf�rica
        positions = project_on_sphere(positions, sphere_center, sphere_radius)
    np.save(f'{outfolder}/equilibrated_filament_N={Numparticles}_fuerza{args.fuerza}_radio_{args.radio}.pkl', positions)

save_vtk(positions, output_file=f'{outfolder}/equilibrated_filament_N={Numparticles}_fuerza{args.fuerza}_radio_{args.radio}.pkl')

factive = factive_old
print(factive)

# Open a file and use dump()
plot_filament_3D(positions, r0=rcut_LJ, output_file=f'{outfolder}/filament_{0}')
save_vtk(positions, output_file=f'{outfolder}/filament_N={Numparticles}_fuerza{args.fuerza}_radio_{args.radio}_t={0}')
for t in (pbar := tqdm(range(1, tsimul + 1))):
    pbar.set_description(f"Step {int}")
    # Calcular fuerzas
    for i in range(0, int(2 / dt)):
        forces = compute_forces_3D(positions, filament_dict, l0)
        # Realizar movimiento browniano
        positions = brownian_motion_3D(positions, forces, dt, gamma, kBT)
        # Proyectar posiciones sobre la superficie esf�rica
        positions = project_on_sphere(positions, sphere_center, sphere_radius)
    # Plot del filamento
    #plot_filament_3D(positions, r0=rcut_LJ, output_file=f'filament_{t}')
    save_vtk(positions, output_file=f'{outfolder}/filament_t={t}')
    
plot_filament_3D(positions, r0=rcut_LJ, output_file=f'{outfolder}/filament_{tsimul + 1}')
np.save(f'{outfolder}/filament_filament_N={Numparticles}_fuerza{args.fuerza}_radio_{args.radio}.pkl', positions)
