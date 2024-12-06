from Bio import PDB
import numpy as np
from sys import argv
from mypy.memprofile import defaultdict
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.spatial import distance

def distance_point_line(point, line_point, line_direction):
    vector_to_point = point - line_point
    projection = np.dot(vector_to_point, line_direction) / np.dot(line_direction, line_direction) * line_direction
    perpendicular_vector = vector_to_point - projection
    distance = np.linalg.norm(perpendicular_vector)
    return distance

def nearest_point_on_line(P0, v, Q):
    w = Q - P0
    t = np.dot(w, v) / np.dot(v, v)
    P = P0 + t * v
    return P

def dihedral_angle_360(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p3
    n1 = np.cross(v1, v2)
    n2 = np.cross(v3, v2)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    dot_product = np.dot(n1, n2)
    angle = np.arccos(np.clip(dot_product, -1, 1))
    sign = np.sign(np.dot(np.cross(n1, n2), v2))
    return np.degrees(angle) * sign

file = argv[1]

# load structure by biopython
structure = PDB.MMCIFParser(QUIET=True).get_structure("structure", file)[0]
nucleic_acid = [res for res in structure.get_residues() if res.resname in {'DA', 'DC', 'DG', 'DT'}]
nucleic_acid_coordinates = np.array([atom.coord for nucleic_acid in nucleic_acid for atom in nucleic_acid.get_atoms()])
protein = [res for res in structure.get_residues() if res.resname not in {'DA', 'DC', 'DG', 'DT'}]
protein_coordinates = np.array([atom.coord for res in protein for atom in res.get_atoms()])

# find central line defined by nucleic acid
pca = PCA(n_components=1)
pca.fit(nucleic_acid_coordinates)
central_line_direction_vector = pca.components_[0]
central_line_mean = pca.mean_


# visualize
fig = go.Figure()
fig.update_layout(title = f"Structure {file}")
fig.add_trace(go.Scatter3d(x = nucleic_acid_coordinates[:,0],
                           y = nucleic_acid_coordinates[:,1],
                           z = nucleic_acid_coordinates[:,2],
                           mode="markers",
                           name="Nucleic acid",
                           marker=dict(size=3,
                                       color="red")))

fig.add_trace(go.Scatter3d(x = protein_coordinates[:,0],
                           y = protein_coordinates[:,1],
                           z = protein_coordinates[:,2],
                           mode="markers",
                           name="Protein",
                           marker=dict(size=3,
                                       color="blue")))
n = 60
fig.add_trace(go.Scatter3d(x=[x*central_line_direction_vector[0] + central_line_mean[0] for x in range(-n, n)],
                           y=[x*central_line_direction_vector[1] + central_line_mean[1] for x in range(-n, n)],
                           z=[x*central_line_direction_vector[2] + central_line_mean[2] for x in range(-n, n)],
                           mode="lines",
                           name="Central line based on nucleic acid"))
fig.show()

# create histogram of distances from central line



distances_protein = []
for res in protein:
    for atom in res.get_atoms():
        distances_protein.append(distance_point_line(atom.coord,
                                                     central_line_mean,
                                                     central_line_direction_vector))
distances_nucleic_acid = []
for res in nucleic_acid:
    for atom in res.get_atoms():
        distances_nucleic_acid.append(distance_point_line(atom.coord,
                                                          central_line_mean,
                                                          central_line_direction_vector))
fig = go.Figure()
fig.update_layout(title=f"Histogram of atoms according their distance from central line",
                  xaxis_title="Distance from central line",
                  yaxis_title="Number of atoms")
fig.add_trace(go.Histogram(x=distances_protein,
                           name="Protein"))
fig.add_trace(go.Histogram(x=distances_nucleic_acid,
                           name="Nucleic acid"))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
radius_covering_95_percent_NA_atoms = sorted(distances_nucleic_acid)[int(len(distances_nucleic_acid)*0.95)]
fig.add_vline(x=radius_covering_95_percent_NA_atoms,
              line_width=2,
              line_dash="dash",
              line_color="green",
              annotation_text=f"  Radius covering 95% nucleic acid atoms: {str(round(radius_covering_95_percent_NA_atoms, 2))}",
              annotation_position="top right")
fig.show()

# create graphs for slices of structure
# all atoms included

# create slices
zero_central_line_nearest_point = nearest_point_on_line(central_line_mean, central_line_direction_vector, [0,0,0])
slices = defaultdict(list)

for atom in structure.get_atoms():
    atom_central_line_nearest_point = nearest_point_on_line(central_line_mean, central_line_direction_vector, atom.coord)
    nearest_points_difference = atom_central_line_nearest_point - zero_central_line_nearest_point
    angle = dihedral_angle_360([0,0,0],
                           zero_central_line_nearest_point,
                           atom_central_line_nearest_point,
                           atom.coord)
    slices[1 * round(angle/1)].append((distance.euclidean(zero_central_line_nearest_point, atom_central_line_nearest_point),
                              distance_point_line(atom.coord,
                                                  central_line_mean,
                                                  central_line_direction_vector)))



# find ideal shift
from scipy import optimize
from scipy.spatial.distance import cdist

def objective_function(shift):
    shifted_x = []
    y = []
    for i, (key, data) in enumerate(sorted(slices.items())):
        shifted_x.extend([x[0]-i*shift[0] for x in data])
        y.extend([x[1] for x in data])
    points = np.array([(a,b) for a,b in zip(shifted_x, y)])
    distances =cdist(points, points)
    total_neares_distances = 0
    for row in distances:
        total_neares_distances += np.sum(np.partition(row, 100)[:100])
    print(shift[0], total_neares_distances)
    return total_neares_distances


optimized_shifts = {"cif_files/8br2.cif": 0.2662628173828124,
                    "cif_files/8bq2.cif": 0.28845153808593743}

if file in optimized_shifts.keys():
    optimized_shift = optimized_shifts[file]
else:
    r = optimize.minimize(objective_function, [0.3], method="Nelder-Mead")
    optimized_shift = r.x[0]


fig = go.Figure()
fig.update_layout(title=f"Protein in reduced dimension",
                  xaxis_title="Relative distance",
                  yaxis_title="Distance from central line")

for i, (key, data) in enumerate(sorted(slices.items())):
    if i == 0:
        showlegend = True
    else:
        showlegend = False
    fig.add_trace(go.Scatter(x=[x[0]-optimized_shift*i for x in data],
                             y=[x[1] for x in data],
                             legendgroup = 1,
                             name=f"{file}, shift={optimized_shift}, total_shift={optimized_shift*360}",
                             showlegend=showlegend,
                             mode="markers",
                             marker=dict(size=10, color="blue")))
fig.show()


# optimalizovat posun a následně překrýt grafy
# udělat válec k kuličkama
# udělat objem za otočku

# https://gitlab.ics.muni.cz/ceitec-cf-biodata/cf-work/-/issues/288


