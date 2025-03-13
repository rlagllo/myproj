import numpy as np
import matplotlib.pyplot as plt
import random
# def monte_carlo(num_samples):
#     x_samples = np.random.uniform(0,1,num_samples)
#     y_samples = np.random.uniform(0,1,num_samples)
#     value = x_samples**2 + y_samples**2
#     return np.mean(value)
# print(monte_carlo(10000))


# def randomwalk_2d(steps):
#     x,y = [0],[0]
#     for _ in range(steps):
#         angle = np.random.uniform(0,2*np.pi)
#         x.append(x[-1]+ np.cos(angle))
#         y.append(y[-1] + np.sin(angle))
#     return x,y
# steps =500
# x,y = randomwalk_2d(steps)

# plt.figure(figsize=(6,6))
# plt.plot(x,y,marker="o",markersize = 2,label = "Random Walk path")
# plt.scatter([0],[0],color = "red", label = "Start")
# plt.legend()
# plt.show()


# def diffusion_1d(num_particles, steps):
#     initial = np.zeros(num_particles)
#     for _ in range (steps):
#         steps = np.random.choice([-1,1],size = num_particles)
#         initial += steps
#     return initial
# num_particles = 1000
# steps = 100
# final_position = diffusion_1d(num_particles,steps)

# plt.hist(final_position, bins = 30, alpha = 0.7,color = "blue",edgecolor = "black")
# plt.xlabel("position")
# plt.ylabel("Number of Particles")
# plt.show()

# def lattice_diffusion(grid_size,steps):
#     grid = np.zeros((grid_size,grid_size))
#     x,y = grid_size // 2, grid_size//2
#     grid[x,y] = 1
#     for _ in range(steps):
#         direction = np.random.choice(["up",'down','left','right'])
#         if direction == 'up' and x>0: x-=1
#         elif direction == 'down' and x<grid_size-1: x+=1
#         elif direction == 'left' and y>0:y-=1
#         elif direction == 'right' and y<grid_size -1:y+=1
#         grid[x,y] +=1
#     return grid
# grid_size = 50
# steps = 1000
# grid = lattice_diffusion(grid_size,steps)
# plt.imshow(grid,cmap="hot",interpolation="nearest")
# plt.title("2D lattice diffution")
# plt.colorbar(label = "number of visits")
# plt.show()

grid_size = 50
num_particles = 100
time_steps = 200
diffusion_bias = 0.6
grid = np.zeros((grid_size,grid_size))

for _ in range(num_particles):
    x,y = grid_size//2, grid_size//2
    grid[x,y] += 1

for t in range(time_steps):
    new_grid = np.zeros_like(grid)
    for x in range(grid_size):
        for y in range(grid_size):
            num_atoms = int(grid[x,y])
            for _ in range(num_atoms):
                direction = random.choices(
                    ['up','down','left','right'],weights = [diffusion_bias,1-diffusion_bias,0.5,0.5]
                )[0]
                if direction == 'up':
                    new_x,new_y = (x-1)%grid_size, y
                elif direction =='down':
                    new_x,new_y = (x+1)%grid_size,y
                elif direction =='left':
                    new_x,new_y = x,(y-1)%grid_size                 
                elif direction =='right':
                    new_x,new_y = x,(y+1)%grid_size   
                new_grid[new_x,new_y] += 1

    grid = new_grid
    if t% 50 == 0 or t == time_steps -1:
        plt.imshow(grid,cmap='hot',interpolation="nearest")
        plt.title(f"Time step: {t}")
        plt.colorbar(label = "Number of atoms")
        plt.show()