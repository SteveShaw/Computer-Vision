# -*- coding: utf-8 -*-

import numpy as np
import fileinput
import matplotlib.pyplot as plt

global x_est
global y_est
global mx
global my

x_est = []
y_est = []
mx = np.array([])
my = np.array([])

dt = 33.0

def kalman_filter(num_obs,S,H,P,F,Q,R,I,cv=True):
    for i in range(num_obs):
    #if i>10:
        #R = np.matrix([[np.std(mx[i-10:i])**2,0.0],[0.0,np.std(my[i-10:i])**2]])
        S = H*S
        P = H*P*H.T + Q   
        V = F*P*F.T + R   
        K = (P*F.T)*np.linalg.pinv(V)
        vx = 0
        vy = 0
        if i>0: 
            vx = (mx[i] - mx[i-1])/dt
            vy = (my[i] - my[i-1])/dt
        if not cv:
            DELTA = np.matrix([mx[i],my[i],vx,vy]).T - (F*S)
        else:
            DELTA = np.matrix([mx[i],my[i]]).T - (F*S)
        S = S + (K*DELTA)
        P = (I - (K*F))*P
        x_est.append(float(S[0]))
        y_est.append(float(S[1]))
    
#constant velocity
def kalman_cv(x_pos, y_pos):
    S = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
    P = 10000.00*np.eye(4)
#P = 0.1
    
    H = np.matrix([[1.0, 0.0, dt, 0.0],\
              [0.0, 1.0, 0.0, dt],\
              [0.0, 0.0, 1.0, 0.0],\
              [0.0, 0.0, 0.0, 1.0]]\
              )
    F = np.matrix([\
        [1.0, 0.0, 0.0, 0.0],\
        [0.0, 1.0, 0.0, 0.0]] \
              )
              
    ra = 1.0**2

    R = np.matrix([[ra,0.0],[0.0,ra]])

    #sv = 1.00

    G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])

    Q = G*G.T
    I = np.eye(4)
    
    kalman_filter(len(x_pos),S,H,P,F,Q,R,I)
    
#constant acceleration
def kalman_ca(x_pos, y_pos):
    S = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P = 10000.00*np.eye(6)
#P = 0.1
    dt = 33.0
    H = np.matrix([[1.0, 0.0, dt, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 1.0, 0.0, dt, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 1.0, 0.0, dt, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    F = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
              
    ra = 1.0**2

    R = np.matrix([[ra, 0.0, 0.0, 0.0],
               [0.0, ra, 0.0, 0.0],
               [0.0, 0.0, 0.01, 0.0],
               [0.0, 0.0, 0.0, 0.01]])

    sv = 0.1

    G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [1.0],
               [1.0]])

    Q = G*G.T*sv
    I = np.eye(6)
    
    kalman_filter(len(x_pos),S,H,P,F,Q,R,I,False)

if __name__=='__main__':
    all_data = [ln.split() for ln in fileinput.input('data.txt')]
    gd_data =  all_data[all_data.index([])+1:] #ground truth
    obs_data = all_data[:all_data.index([])]
    x_pos = [float(item[0]) for item in obs_data]
    y_pos = [float(item[1]) for item in obs_data]
    gdx_pos = [float(item[0]) for item in gd_data]
    gdy_pos = [float(item[1]) for item in gd_data]
    mx = np.array(x_pos)
    my = np.array(y_pos)
    kalman_ca(mx,my)
    
    #plot ground truth and estimation
    plt.plot(x_est,y_est,'ro',label='Kalman Estimate')
    gd_plot, = plt.plot(gdx_pos,gdy_pos,'g-',label='Ground Truth')
    obs_plot, = plt.plot(x_pos,y_pos,'b',label='Observe')
    #plt.legend(est_plot,gd_plot,obs_plot],['Kalman Estimate','Ground Truth','Observe'])
    plt.legend(loc='upper left',fontsize='small',title='constant acceleration model')
    #plt.show()


#S = np.matrix([[x_pos[0], y_pos[0], 0.0, 0.0]]).T




        
 

