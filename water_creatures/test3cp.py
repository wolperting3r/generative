import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.box(False)
import os
# from progressbar import *
os.system('rm *.png')

n_points = 20
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

export = True
if export:
    matplotlib.use('macosx')

fadeout = 50
its = 3000
spe = 500
start = 150
plot = False
for nn in range(10):
    print(f'nn:\n{nn}')
    # widgets = ['Generating: ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA(), '']
    # pbar = ProgressBar(widgets=widgets, maxval=(its-start))
    for it in range(its):
        if it == start:
            print('Plot')
            plot = True
            '''
            lims = [
                min(np.amin(points[:, 0], 0), np.amin(points[:, 1]))-2,
                max(np.amax(points[:, 0], 0), np.amax(points[:, 1]))+2
            ]
            # '''
            lims_x = np.array([np.amin(points[:, 0])-1, np.amax(points[:, 0])+1])
            lims_y = np.array([np.amin(points[:, 1])-1, np.amax(points[:, 1])+1])
            # mid_0 = np.sum(points, axis=0)/n_points
            # mid = (np.sum(points, axis=0)/n_points - mid_0)**1
            # pbar.start()
        if plot:
            # pbar.update(it-start)
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.axis('off')
            ax.set_frame_on(False)
            ax.set_facecolor((60/256, 7/256, 86/256))
            # m_al = 0.5
            # mid = m_al*((np.sum(points, axis=0)/n_points - mid_0)**1) + (1-m_al)*mid  # 
            mid = np.sum(points, axis=0)/n_points
            l_al = 0.01  # 0.1
            lims_x = l_al*np.array([np.amin(points[:, 0])-2, np.amax(points[:, 0])+2]) + (1-l_al)*lims_x
            lims_y = l_al*np.array([np.amin(points[:, 1])-2, np.amax(points[:, 1])+2]) + (1-l_al)*lims_y
            ax.set_xlim(lims_x)
            ax.set_ylim(lims_y)
            # ax.scatter(points[:,0], points[:,1], color='k', s=5)

        force[:] = 0
        for ind, p in enumerate(points):
            # print(f'p:\n{p}')
            dis_vec = p - points
            dis = np.ma.array(dis_vec[:, 0]**2 + dis_vec[:, 1]**2, mask=False)
            dis.mask[ind] = True
            if it > (start-10):
                for i in range(n_points):
                    near_ind = np.argmin(dis)
                    alpha = (0.05 if it > 100 else 0.1)  # 0.017  # 0.025
                    linewidth[ind, near_ind] = max(alpha * min((0.6*linewidth[ind, near_ind]+0.7)/(0.1*i**2+1) - 0.24/(i+1), 1) + (1-alpha)*linewidth[ind, near_ind], 0)
                    # linewidth[ind, near_ind] = max(alpha * min(1/(0.1*i**2+1) - 0.24/(i+1), 1) + (1-alpha)*linewidth[ind, near_ind], 0)
                    if it > its-fadeout:
                        linewidth[ind, near_ind] = linewidth[ind, near_ind] - 3*(it/its)**5/fadeout*linewidth[ind, near_ind] + min(np.cos(20*it/its)+mv1[ind][0], 1)/fadeout - 1/fadeout
                    # *(1/(dis[near_ind]*0.1))
                    # 0.4 * max((i+1)/3, 0)*min(1/dis[near_ind], 1) + 0.6*linewidth[ind, near_ind]
                    dis.mask[near_ind] = True
                    # '''
            con = 4
            lwds = np.ma.array(linewidth[ind, :], mask=False)
            if plot:
                for i in range(con):
                    paint_ind = np.argmax(lwds)


                    lwds.mask[paint_ind] = True
                    nep = np.reshape(points[paint_ind], (2,))
                    # '''
                    plt.plot([p[0], nep[0]], [p[1], nep[1]],
                             color='k',  # 'lightyellow',
                             linewidth= max(min(8*(linewidth[ind, paint_ind]**(3)-1e-5), 1), 0), # 1*linewidth[ind, paint_ind],  # (min(1/dis[near_ind],1)),
                             alpha= max(min(3*(linewidth[ind, paint_ind]**(5)-2e-3), 1), 0)
                             )
                    # '''
                    dis.mask[near_ind] = True

            freq=6

            force_vec_1 = np.nan_to_num(
                (-0.2*np.cos((freq*it/spe)*np.pi)-0.30)
                * np.divide(
                    dis_vec,
                    np.reshape(dis_vec[:, 0]**2 + dis_vec[:, 1]**2 + 1e-7, (n_points, 1))
                ))

            force_vec_2 = np.nan_to_num(
                (0.2*np.sin((freq+1)*it/spe*np.pi+0.3)+0.3)  # 0.3
                * np.divide(
                    dis_vec,
                    np.reshape((dis_vec[:, 0]**2 + dis_vec[:, 1]**2 + 1e-7)**2, (n_points, 1))
                ))


            force[ind] = force[ind] + np.sum(force_vec_1, axis=0) + (np.sum(force_vec_2, axis=0) if it < its-fadeout else 0)
            # + 50*(np.random.rand(1,2)*speed-speed/2)


        # f_tan = [2, 10]      # zu stark
        # f_tan = [0.2, 0.1]      # stark  -> mv2 * 1
        f_tan = [0.05, 0.025]   # schwach  -> mv2 * 0.2

        force = 0.8*force + mv2*0.2*np.sin(1.5*freq*it/spe*np.pi) + 1*(np.random.rand(n_points, 2)-0.5)*(0.3-vel) 
        if it < its-fadeout:
            force = force + mv1*f_tan[0]*(min(np.tan(0.9*freq*it/spe*np.pi), 3))+ mv1*f_tan[1]*(min(np.tan(0.9*freq*it/spe*np.pi+0.3), 3))

        r_force = np.zeros((n_points, 2))
        if plot:
            # b_force = np.divide(points - mid, [(lims_x[1]-lims_x[0])/1.7, (lims_y[1]-lims_y[0])/1.7])
            b_force = -np.array([1/(points[:, 0] - lims_x[1]) + 1/(points[:, 0] - lims_x[0]), 1/(points[:, 1] - lims_y[1]) + 1/(points[:, 1] - lims_y[0])]).T
            b_force = np.power(b_force, 5)
            b_force[np.nonzero(np.abs(b_force)< 0.1)] = 0


            r_force = np.power(points-mid, 3)
            mid_dis = np.sum(np.abs(points-mid))/n_points
            r_force[np.nonzero(np.abs(r_force)< 12*mid_dis)] = 0
            # r_force[:, 0], r_force[:, 1] = -r_force[:, 1], r_force[:, 0].copy()  # rotation
            force = force-b_force-0.1*r_force

            '''
            # plt.scatter(points[:,0], points[:,1], color='k', s=5)
            # print(f'r:\t{np.amax(np.abs(r_force), axis=0)},\tb:\t{np.amax(np.abs(b_force), axis=0)}')
            printpoints = points[np.nonzero(np.abs(np.sum(r_force, axis=1) + np.sum(b_force, axis=1)) > 0)]
            if len(printpoints) > 0:
                plt.scatter(printpoints[:,0], printpoints[:,1], color='r', s=20)
            # for ppid, pp in enumerate(printpoints):
                # print(f'b_force[0].shape:\n{b_force[0].shape}')
                # plt.plot([pp[0], pp[0] + 0.1*r_force[ppid, 0]], [pp[1], pp[1] + 0.1*r_force[ppid, 1]], color='r')
            # '''

        # vel[2] = vel[2] + 0.05*(np.random.rand()-0.5)*np.arctan(1/(vel[2]+1e-5))*force[2]
        # vel[4] = vel[4] + 0.1*(np.random.rand()-0.5)*np.arctan(1/(vel[2]+1e-5))
        # vel[3] = vel[3] + np.array([0.2, 0.2])
        vel = np.clip(vel + 0.1 * force - 0.4*vel, a_max=3, a_min=-3)  # + 0.1*np.random.rand(n_points, 2)
        points = points + vel*0.1

        if plot:
            if export:
                fig.tight_layout()
                fig.savefig(f'{it:04d}.png', dpi=100)
            else:
                plt.pause(0.001)
            plt.close()

        '''
        fst = 1 - (it-1*its/nx)/(its/nx)*np.sign(it-1*its/nx+1e-10)
        fst = fst if fst > 0 else 0
        sec = 1 - (it-2*its/nx)/(its/nx)*np.sign(it-2*its/nx+1e-10)
        sec = sec if sec > 0 else 0
        thr = 1 - (it-3*its/nx)/(its/nx)*np.sign(it-3*its/nx+1e-10)
        thr = thr if thr > 0 else 0
        # '''
    #plt.show()

    # pbar.finish()
    np.savetxt('points.txt', points)
    if export:
        os.system(f'convert -delay 1x30 -loop 0 *.png result_{nn}.gif')
    os.system('rm *.png')
