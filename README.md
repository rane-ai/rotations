# rotations

A simple implementation of ICP demonstrating different rigid body parametersations. The code has been written 
for clarity/ease-of-understanding and is definitely not optimal! The point to point distance between 2 (roughly 
aligned) scans is minimised using Gauss Newton. The approach described in [0] was used to optimise the relative 
pose. Three parameterisations for rigid body increments delta_X are considered:

(1) Euler :
        delta_X = [delta_x, delta_y, delta_z, delta_phi, delta_gamma, delta_psi]

(2) Twist (linear/angular velocities):
        delta_X = [u_x, u_y, u_z, omega_x, omega_y, omega_z]

(3) Quaternion:
        delta_X = [delta_x, delta_y, delta_z, q_x, q_y, q_z]

There is a lot of room for improvement e.g. point to plane metric, solver etc.

[0] "Least Squares Optimization: from Theory to Practice" by Grisetti et al https://arxiv.org/pdf/2002.11051.pdf


# setup
```
python -m venv env
source env/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
```