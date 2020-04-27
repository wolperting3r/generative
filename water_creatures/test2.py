import numpy as np
import matplotlib.pyplot as plt
# import os
import time

n_points = 25
points = np.random.rand(n_points, 2)*10
# points = np.loadtxt('points.txt'); n_points=100
force = np.zeros((n_points, 2))
vel = np.zeros((n_points, 2))

linewidth = np.zeros((n_points, n_points))

speed = 3
nx = 4
mv1 = np.random.rand(n_points, 2)*speed-speed/2
mv2 = np.random.rand(n_points, 2)*speed-speed/2
mv3 = -mv1 - mv2

its = 500
for it in range(its):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax.set_facecolor((60/256, 7/256, 86/256))
    # Make everything except the plot white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    lims = [-4, 14]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    time0 = time.time()

    time1 = time.time()
    force[:] = 0
    # plt.scatter(points[:,0], points[:,1], color='lightyellow', s=5)
    for ind, p in enumerate(points):
        # print(f'p:\n{p}')
        dis_vec = p - points
        dis = np.ma.array(dis_vec[:, 0]**2 + dis_vec[:, 1]**2, mask=False)
        dis.mask[ind] = True
        # for i in range(int(min(n_points/2, 5))):
        for i in range(n_points):
            near_ind = np.argmin(dis)
            alpha = 0.1
            linewidth[ind, near_ind] = max(alpha * min(1/(0.1*i**2+1) - 0.1/(i+3), 1) + (1-alpha)*linewidth[ind, near_ind], 0)
            # *(1/(dis[near_ind]*0.1))
            # 0.4 * max((i+1)/3, 0)*min(1/dis[near_ind], 1) + 0.6*linewidth[ind, near_ind]
            dis.mask[near_ind] = True
            # '''
        con = 3
        lwds = np.ma.array(linewidth[ind, :], mask=False)
        for i in range(con):
            paint_ind = np.argmax(lwds)
            lwds.mask[paint_ind] = True
            nep = np.reshape(points[paint_ind], (2,))
            plt.plot([p[0], nep[0]], [p[1], nep[1]],
                     color='k',  # 'lightyellow',
                     linewidth=linewidth[ind, paint_ind],  # (min(1/dis[near_ind],1)),
                     alpha=max(min(5*(linewidth[ind, paint_ind]**(8)-1e-3), 1), 0)
                     )
            # '''
            dis.mask[near_ind] = True

        force_vec_1 = np.nan_to_num(
            (-0.2*np.cos((30*it/its)*np.pi)-0.1)
            * np.divide(
                dis_vec,
                np.reshape(dis_vec[:, 0]**2 + dis_vec[:, 1]**2, (n_points, 1))
            ))

        force_vec_2 = np.nan_to_num(
            (0.5*np.sin(31*it/its*np.pi+0.3)+0.5)
            * np.divide(
                dis_vec,
                np.reshape((dis_vec[:, 0]**2 + dis_vec[:, 1]**2)**2, (n_points, 1))
            ))

        force[ind] = force[ind] + np.sum(force_vec_1, axis=0) + np.sum(force_vec_2, axis=0)
        # + 50*(np.random.rand(1,2)*speed-speed/2)

    force = force + mv1*0.8*np.sin(20*it/its*np.pi) + 2*(np.random.rand(n_points, 2)-0.5)*(0.3-vel)  # *np.cos(40*it/its*np.pi)

    plt.pause(0.005)

    # fig.tight_layout()
    # fig.savefig(f'{it:04d}.png', dpi=100)
    plt.close()

    vel = vel + 0.1 * force - 0.1*vel

    '''
    fst = 1 - (it-1*its/nx)/(its/nx)*np.sign(it-1*its/nx+1e-10)
    fst = fst if fst > 0 else 0
    sec = 1 - (it-2*its/nx)/(its/nx)*np.sign(it-2*its/nx+1e-10)
    sec = sec if sec > 0 else 0
    thr = 1 - (it-3*its/nx)/(its/nx)*np.sign(it-3*its/nx+1e-10)
    thr = thr if thr > 0 else 0
    # '''
    # points = points + mv1*fst + mv2*sec + mv3*thr
    points = points + vel*0.1

    # np.savetxt('points.txt', points)
    # Insert Gravity = distance

plt.show()


# os.system('convert -delay 1 -loop 0 *.png result.gif')
