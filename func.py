
import pandas as pd
import numpy as np, time, os
from scipy import stats

def image_binary2(image, threshold = 1.15):
    image = image.astype(float)

    temp = 1* (2* image[:, :, 1]/(image[:, :, 0] + image[:, :, 2]) > threshold)
    return temp


# directory: soybean directory under path
def get_pixel_from_sub_directory(path_of_directory, directory_list, sub_dir_dic, 
                                  threshold, save_image = False,
                                 save_file= True, file_name= 'Temporary.csv', 
                                 is_concat = False, df1 = pd.DataFrame()):
    count=0
   
    thre_dic={'threshold':[]}
    for directory in directory_list:
        os.chdir(path_of_directory + directory)
        thre_dic['threshold'].append(threshold)
        #print(directory)
        #camera_list: a list of camera directory under path + directory
        camera_list = [camera for camera in os.listdir(path_of_directory + directory ) if (os.path.isdir('.'+ '\\'+camera))]
        #print(camera_list)
       
        # append number of png files in each camera directory. if there is no camera directory on the path+directory, append NA
        sub_count = 0
        bgr_img = None
        for i, key in enumerate(sub_dir_dic):   
            
            if key in camera_list:
                os.chdir(path_of_directory + directory +'\\'+ key)
                filenames = [file for file in glob.glob('*.png') if 'rescaled' not in file]
                bgr_img = cv2.imread(filenames[0]).astype(float)
  
                crt=image_binary2(bgr_img,  threshold = threshold)
                if save_image:
                    
                    cv2.imwrite(str(thre) + '_' + filenames[0], crt)
   
                sub_dir_dic[key].append((crt==1).sum())
            else:
                sub_dir_dic[key].append('NA')
            count= count+1
            
            print(count)
        
            if save_file:
                if i == len(sub_dir_dic)-1 :
                    print('saving file for '+ str(count)+ ' plant images.')
                    df2=pd.DataFrame.from_dict(sub_dir_dic)
                
                    if is_concat:
                        df= pd.concat([df1, df2], axis=1)
                        df.to_csv(path_of_directory+file_name, index=False)
                    
                    else:
                        df2.to_csv(path_of_directory+ file_name, index=False)
   
    df2=pd.DataFrame.from_dict(sub_dir_dic)
    if is_concat:
        df= pd.concat([df1, df2], axis=1)
        df.to_csv(path_of_directory+file_name, index=False)

    else:
        df = df2
        df.to_csv(path_of_directory+ file_name, index=False)
    return thre_dic, df2, df
    
   

        

def admm_solver(X, y, p, lamb = 0, max_iter = 1000):

    '''
    Output:
    beta estimator
    '''
    rho = 10**11
    

    beta_t = []
    #beta_t = np.random.randn(p, 1)
    #F = np.random.randn(n, p)
    #print(beta_t)
    kappa_t =np.zeros((p,1))
    
    u_t = np.zeros((p,1))
    
    #c_t = np.zeros(max_iter)

    mu, tau_incr, tau_decr = 10, 2, 2
    
    for i in range(max_iter):
        
        #c_t, F= gradient_descent(beta_t,  y, v, n, p, scale = 1, max_iter = 100)

        P = X.T.dot(X) + rho * np.identity(p)
        #print(F.shape)
        #print(u_t.shape)
        #print(kappa_t.shape)
        q = -X.T.dot(y) + rho*(u_t - kappa_t)
        beta_t.append(np.linalg.inv(P).dot(-q))
        
        #beta_t.append(admm_qp(P, q, outer_iter = i, max_iter = 10))
        #beta_t.append(true_beta)
        #beta_t =np.array([0, 4.85455, 0.321547, 0.218930, 0]).reshape(5, 1)

        #print(beta_t)

        kappa_t_ = kappa_t
       
        kappa_t = -lamb/rho+ beta_t[i] + u_t
        #print('kappa ', kappa_t)
        kappa_t = np.where(kappa_t >= 0, kappa_t, 0)
        
        #c_t[i], F= gradient_descent(beta_t[i],  y, v, n, p, scale = scale, max_iter = 1000)
        
        #c_t = 0.8
        #F, _ = cal_F_D(n, p, v, c_t, scale =scale)
        
 
        u_t = u_t + beta_t[i] - kappa_t

        r_t = beta_t[i] -kappa_t
        s_t = rho*(kappa_t -kappa_t_)
        """
        if sum(np.power(r_t, 2)) > mu * sum(np.power(s_t, 2)) :
            rho =tau_incr*rho
        elif sum(np.power(s_t, 2))  > mu * sum(np.power(r_t, 2)) :
            rho = rho/tau_decr
        """
        if i>1 and sum(abs(beta_t[i]-beta_t[i-1]))<10**(-10):
            #print('number of iteration', i)
            break
        '''
        if i%100 == 0:
            print(i)
            cost = (1/2) * (X.dot(beta_t[i])- y).T.dot(X.dot(beta_t[i])- y) + lamb* sum(kappa_t) +             (1/2)*rho*(beta_t[i]-kappa_t + u_t).T.dot(beta_t[i]-kappa_t + u_t)-(1/2)*rho*u_t.T.dot(u_t)
            print('cost function', cost)
            #print('c', c_t[i])
            print('beta', beta_t[i])
            print('primal residual r_t: ', r_t)
            print('dual residual s_t: ', s_t)
            print('kappa: ', kappa_t)
        #if c_t ==False:
         #   return F, _
         '''
        #print('Does not converge')
        if i == max_iter-1:
            print('Does not converge')
    return beta_t



def msve_comparison(X_train, y_train, X_test, y_test, alpha_grid, n_iter ):
    """
    Given ADMM results, Perform cross validation to select lambda, the penality parameter for L1 norm
    Output:
    mean square validation error for a given ADMM based algorithm results
    """

    kf5 = KFold(n_splits=5, shuffle=False)
    p = X_train.shape[1]
    cost = lambda y_true, y_pred, lamb, beta: \
            (1/2) * np.sum((y_true - y_pred) ** 2)+ lamb*np.linalg.norm((beta), ord=1)
    mse=[]
    msve=[]
   # mspe=[]
    for alph in alpha_grid:
   
        #mse_fold = []
        msve_fold=[]
        #mspe_fold=[]

        for train_index, valid_index in kf5.split(X_train):


            X_train_fold=X_train.iloc[train_index].to_numpy(copy=True)
            X_valid_fold=X_train.iloc[valid_index].to_numpy(copy=True)

            y_train_fold=y_train.iloc[train_index].to_numpy(copy=True).reshape(-1, 1)
            y_valid_fold=y_train.iloc[valid_index].to_numpy(copy=True).reshape(-1, 1)

            
            
            
            #X_train_fold=X_train.iloc[train_index]
            #X_valid_fold=X_train.iloc[valid_index]

            #y_train_fold=y_train.iloc[train_index]
            #y_valid_fold=y_train.iloc[valid_index]
            n = X_train_fold.shape[0]
            
            
            #X_train_np = X_train_fold.to_numpy(copy=True)
            #y_train_np = y_train_fold.to_numpy(copy=True).reshape(-1, 1)
            ##normalized X
            #X_train_np = (X_train_np-np.min(X_train_np, axis=0))/(np.max(X_train_np, axis=0) - np.min(X_train_np, axis=0))
            lamb = n*alph
            beta_t = admm_solver(X_train_fold, y_train_fold, p, lamb= lamb , max_iter=n_iter)
            #beta_t = admm_solver(X_train_np, y_train_np, p, lamb= lamb , max_iter=n_iter)
            y_valid_pred = X_valid_fold.dot(beta_t[-1])
            y_train_pred = X_train_fold.dot(beta_t[-1])
            y_test_pred = X_test.to_numpy().dot(beta_t[-1])
            
     
            #mse_fold.append(cost(y_train_fold, y_train_pred, lamb, beta_t[-1]))
            #print(cost(y_train_fold, y_train_pred, lamb, beta_t[-1]))
            #msve_fold.append(cost(y_valid_fold, y_valid_pred, lamb, beta_t[-1]))
            msve_fold.append(mean_squared_error(y_valid_pred, y_valid_fold))
            #print(cost(y_valid_fold, y_valid_pred, lamb, beta_t[-1]).shape)
         
            #print('cost', cost(y_valid_fold, y_valid_pred, lamb, beta_t[-1]))
    
            #print('\n')
            #print('msve_fold', msve_fold)
            #mspe_fold.append(cost(y_test, y_test_pred, lamb, beta_t[-1]))
            
        #mse.append(np.mean(mse_fold))
        msve.append(np.mean(msve_fold))
        #mspe.append(np.mean(mspe_fold))
        #print(msve)
    return msve
   # return msve
   
   
   

def sim_control_cusum(x, theta, target, tol):
    """ This function calculate cusum statistics for sorted_dataframe[data_key] column, for each id_key, 
    return the cusum threshold info for each delta and include cusum statistics in sorted_dataframe

    sorted_dataframe: input sorted_dataframe that contains all information, sort by id_key and measure time
    id_key: groupby value, calculate cusum for each id object
    delta: max/min(data_avg, delta), quantify the magnitude of change
    target: target length of control group to determine the threshold value for treatment group
    tol: target +- tol
    """
    start_time= time.time()
    
    control_result_dic = {}
    
    #if sorted_dataframe == pd.DataFrame()

   

    for the in theta[:]:
        print(the)
        
        cusum_list = []
        #cusum = np.zeros(sorted_dataframe.shape[0])
        count = 0
        for subdata in x:
            if count % 500 ==0:
                print(count)
            count = count + 1
            cusum_temp = [0]

            for i in range(len(subdata)):
                        
                logLR = the*(subdata[i] - the/2)
                cusum_temp.append(max(cusum_temp[-1] + logLR, 0))
                    
            cusum_list.append(cusum_temp[1: ])
            
        
        cusum = [item for items in cusum_list for item in items]
        #cusum = cusum_arr.flatten()
 
        #cusum_dic[the] =cusum


        b, N = bisection_search(cusum_list, target, tol)

        ## Output dictionaries 3    
        control_result_dic[the] ={}
        control_result_dic[the]['b'] = b
        control_result_dic[the]['mean'] = round(np.mean(N), 4) 
        control_result_dic[the]['se'] = round(stats.sem(N), 4)
        control_result_dic[the]['N'] = N
        

        end_time= time.time()
        print("--- %s seconds ---" % (end_time - start_time))
    return control_result_dic


  
    
def treatment_cusum(sorted_dataframe, id_key, data_key, theta, control_result_dic):

    """ This function calculate cusum statistics for sorted_dataframe[data_key] column, for each id_key, 
    return the cusum threshold info for each delta and include cusum statistics in sorted_dataframe

    sorted_dataframe: input sorted_dataframe that contains all information, sort by id_key and measure time
    id_key: groupby value, calculate cusum for each id object
    delta: max/min(data_avg, delta), quantify the magnitude of change
    control_result_dic: output from control_cusum that contains threshold information
    """
    start_time= time.time()
    x = sorted_dataframe.groupby(id_key)[data_key].apply(pd.Series.tolist).tolist()
 
    treatment_result_dic = {}
    for the in theta[:]:
        print(the)
             
            
        cusum_list = []
        #cusum =[]



        N = []
        #sorted_dataframe = sorted_dataframe.sort_values([id_key, 'Lag']) 
        count = 0

        for subdata in x:
            
            if count % 500 ==0:
                print(count)
            count = count + 1

            cusum_temp =[0]
            record = True

            for i in range(len(subdata)):


                logLR = the*(subdata[i] - the/2)
                cusum_temp.append(max(cusum_temp[-1] + logLR, 0))

                if record and (cusum_temp[-1] > control_result_dic[the]['b'] or i == len(subdata) -1):
                    N.append(i)
                    record = False
                    
            cusum_list.append(cusum_temp[1:])
                    
        cusum = [item for items in cusum_list for item in items]
        sorted_dataframe[the] = cusum

        treatment_result_dic[the] ={}
        treatment_result_dic[the]['b'] = control_result_dic[the]['b']

        # mean and se, average length of change time and the standard error based on the threshold value b
        treatment_result_dic[the]['mean'] = round(np.mean(N), 4) 
        treatment_result_dic[the]['se'] = round(stats.sem(N), 4)
        treatment_result_dic[the]['N'] = N

        end_time= time.time()
        print("--- %s seconds ---" % (end_time - start_time))    
    return treatment_result_dic, sorted_dataframe

def treatment_adaptive_cusum(sorted_dataframe, id_key, data_key, delta, control_result_dic):

    """ This function calculate cusum statistics for sorted_dataframe[data_key] column, for each id_key, 
    return the cusum threshold info for each delta and include cusum statistics in sorted_dataframe

    sorted_dataframe: input sorted_dataframe that contains all information, sort by id_key and measure time
    id_key: groupby value, calculate cusum for each id object
    delta: max/min(data_avg, delta), quantify the magnitude of change
    control_result_dic: output from control_cusum that contains threshold information
    """
    start_time= time.time()
    x = sorted_dataframe.groupby(id_key)[data_key].apply(pd.Series.tolist).tolist()
    t = 1

    treatment_result_dic = {}
    for dta in delta[:]:
        print(dta)
        s = t*dta
        
        if dta > 0:                
            #s = 1
            func = max
        else:
            #s = -1
            func = min
            
            
        cusum_list = []
        #cusum =[]
        avg =[]
    
        N = []
        change_N = []      
        #sorted_dataframe = sorted_dataframe.sort_values([id_key, 'Lag']) 
        count = 0
        exclude_add = 0

        for subdata in x:
            if count % 500 ==0:
                print(count)
            count = count + 1
            s_n = 0
            t_n = 0
            mu_temp =[]
            cusum_temp =[0]
            need_record = True
            is_change = False
            for i in range(len(subdata)):
                
                if cusum_temp[-1] ==0:
                     #s_n = subdata[data_key].iloc[i-1]
                    #t_n = 1                       
                    s_n = 0
                    t_n = 0
                        
                else:
                    s_n = s_n + subdata[i-1]
                    t_n = t_n + 1
                        #print(subdata[data_key].iloc[i-1])
                        #print(s_n)
                        #print(t_n)
                        
                mu_temp.append(func((s_n+s)/(t_n + t), dta))   
                avg.append((s_n+s)/(t_n + t))


                logLR = mu_temp[i]*(subdata[i] - mu_temp[i]/2)
                cusum_temp.append(max(cusum_temp[-1] + logLR, 0))
                    #cusum.append(max(cusum[-1] + logLR, 0))
                    
                if cusum_temp[-1] == 0 and need_record:
                    is_change= False
                    
                if (is_change == False) and need_record and (cusum_temp[-1] > 0):# or i == len(subdata) -1):
                    change_time = i
                    #print('change_time', i)
                    is_change = True
                    
                if (is_change == False) and need_record and i == len(subdata) -1:
                    change_time = np.nan
                    #print('change_time', i)
                    is_change = True
                    exclude_add += 1

                if need_record and (cusum_temp[-1] > control_result_dic[dta]['b'] or i == len(subdata) -1):
                    N.append(i)
                    #print('need_record_time', i)
                    #print('length', len(subdata))

                    need_record = False     


                    
            cusum_list.append(cusum_temp[1:])
            change_N.append(change_time)
            #if N[-1] == change_N[-1]:
             #   exclude_add += 1
                    
        cusum = [item for items in cusum_list for item in items]
        sorted_dataframe[dta] = cusum
        sorted_dataframe[str(dta) + '_bar'] = avg


        treatment_result_dic[dta] ={}
        treatment_result_dic[dta]['b'] = control_result_dic[dta]['b']

        # mean and se, average length of is_change time and the standard error based on the threshold value b
        treatment_result_dic[dta]['mean'] = round(np.mean(N), 4) 
        treatment_result_dic[dta]['se'] = round(stats.sem(N), 4)
        treatment_result_dic[dta]['N'] = N
        treatment_result_dic[dta]['change_N'] = change_N

        end_time= time.time()
        print(48- exclude_add)
        print("--- %s seconds ---" % (end_time - start_time))    
    return treatment_result_dic, sorted_dataframe
	
	
	



def bisection_search(data_list, target, tol):
    """ This function implement binary search algorithm to find a threshold value b that satisfied these conditions:
    N = [index of first value in list in data_list that list > b]
    mean of N approx to target value within target +/- tol

    data_list: (cusum_list) list of lists contains numbers
    target+/-tol: range of mean value that want to achieve
    return:
    b: the threshold value
    N: a list, it contains the index+ 1 of first value in list that in data_list. 
    

    """

    lower = 0
    upper = 500
    b = 0.5*(lower + upper)
    while True:

        print(b)

        N = []

        for c in data_list:
        

            indices = [ind for ind,  ci in enumerate(c) if ci> b]

            if indices:
                N.append(indices[0])

            else:
                N.append(len(c)-1)
                #N.append(find_value(arr = c >b, value = True))
                            #print(len(ai_arr))


        N_mean = np.mean(N) + 1
        if abs(N_mean- target) <tol:
            break

        elif N_mean < target:
            lower = b
            b= 0.5*(b + upper)

        elif N_mean > target:
            upper = b
            b = 0.5*(lower + b)
    return b, N


def sim_control_adaptive_cusum(x, delta, target, tol):
    """ Used in simulation or input is a np.array with equal length per data stream.  
    This function calculate cusum statistics for sorted_dataframe[data_key] column, for each id_key, 
    return the cusum threshold info for each delta and include cusum statistics in sorted_dataframe

    x: input data
    m, n: m*n dimension of the data. m: # data stream, n: length of each data stream
    delta: max/min(data_avg, delta), quantify the magnitude of change
    target: target of control group to determine the threshold value for treatment group
    tol: target +- tol
    """


    start_time= time.time()
    control_result_dic = {}
    t = 1


    for dta in delta[:]:
        print(dta)
        
        s = t*dta
        if dta > 0:                
            #s = 1
            func = max
        else:
            #s = -1
            func = min

        
        #s_n, t_n, mu_temp, cusum = np.zeros((m, n_max)), np.zeros((m, n_max)), np.zeros((m, n_max)), np.zeros((m, n_max))
        #mu_temp[:, 0] = s/t

        #cusum[:, 0] = np.max(np.append(mu_temp[:, 0]*(x[:, 0] + mu_temp[:, 0]/2), 0))
        cusum_list = []
        #cusum = np.zeros(sorted_dataframe.shape[0])
        
       
        count = 0
            #spline_control = spline_control.sort_values(['Plant ID', 'Lag'])

        for subdata in x:
            if count % 500 ==0:
                print(count)
            count = count + 1
            
            mu_temp =[]
            cusum_temp = [0]
            s_n = 0
            t_n = 0

            for i in range(len(subdata)):

                if cusum_temp[-1] == 0:
                        #s_n = subdata[data_key].iloc[i-1]
                        #t_n = 1                       
                    s_n = 0
                    t_n = 0

                else:
                    s_n = s_n + subdata[i-1]
                        #print(s_n)
                    t_n = t_n + 1
                        
                mu_temp.append(func((s_n+s)/(t_n + t), dta))
                logLR = mu_temp[i]*(subdata[i] - mu_temp[i]/2)
                cusum_temp.append(max(cusum_temp[-1] + logLR, 0))
                    
            cusum_list.append(cusum_temp[1: ])
                ## Output dictionaries 1, 2
            #control_cusum_dic[dta][p_id]= cusum_temp
            #mu_dic[dta][p_id] = mu_temp
        #sorted_dataframe[dta] = cusum

        b, N = bisection_search(cusum_list, target, tol)
        

        ## Output dictionaries 3    
        control_result_dic[dta] ={}
        control_result_dic[dta]['b'] = b
        control_result_dic[dta]['mean'] = round(np.mean(N), 4) 
        control_result_dic[dta]['se'] = round(stats.sem(N), 4)
        control_result_dic[dta]['N'] = N
		
        end_time= time.time()
        print("--- %s seconds ---" % (end_time - start_time))

    return control_result_dic


def dic_index_to_alarm_time(col_list, result_dic, id_list, sorted_lag_list):
    """ This function calculates the raise alarm time for each object id based on the cusum statistics
    col_list: a list that contains the column names shared in result_dic. 
    eg. delta value used in control/treatment_cusum
    It returns a dataframe that contains alarm time information

    treatment_result_dic: a dictionary that contains index list for each cusum column name. eg {dta: {'b': [index Ns]}}
    id_list: id for each subject, used for index for output dataframe
    sorted_lag_list: sorted_lag_list[index] = Lag time
    """
    alarm_time = pd.DataFrame(index = id_list)

    for dta in col_list[:]:
        col_name = '_'.join([str(dta), 'Alarm'])
        alarm_time[col_name] = [sorted_lag_list[index] for index in result_dic[dta]['N']]   
    return alarm_time