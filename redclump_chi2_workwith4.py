import  pickle
import pyfits 

stars_scaled_errs = zip(feherr,cerr,nerr,oerr,naerr,mgerr,alerr,sierr,serr, kerr, caerr, tierr,   verr, mnerr,nierr,perr, crerr,coerr, cuerr,rberr)
stars_scaled= zip(feh,c,n,o,na,mg,al,si,s, k, ca, ti,   v, mn,ni,p, cr,co, cu,rb)
stars_scaled =  zip(feh_rs,c_rs,n_rs,o_rs,na_rs,mg_rs,al_rs,si_rs,s_rs, k_rs, ca_rs, ti_rs,   v_rs, mn_rs,ni_rs,p_rs, cr_rs,co_rs, cu_rs,rb_rs)


chi2_all = [] 
chi2_elem_all = [] 
ids_all = [] 
a = open('redclumpall123_covs.pickle', 'r')
covs = pickle.load(a)
covs2 = covs[:,2:, 2:] 
cinv1 = np.linalg.inv(covs2) 
cinv = (cinv1*cinv1)**0.5

chi2_all = []
ids_all= [] 
getchi2(stars_scaled, cinv, feh) 
def getchi2(elem_array, error_matrix,elem1): 
  counts = arange(0,len(elem1),1) 
  # INPUTS 
  # elem_array of shape 1 x N (N = number of elements)
  # error matrix is the number of stars by the number of elements by the number of elements: M x N x N 
  # elem1 is the first element =  feh   which is of size 1 x M where M is the number of stars
  # below go through for each star and get the chi2 - sorting to compare to just nearest neighbours in fe/h 

  # argsort so do chi2 just on nearest
  sort1 = argsort(elem1) 
  stars_scaled = array(stars_scaled)
  stars_scaled_sort1 = stars_scaled[sort1]
  stars_id_sort1 = counts[sort1]
  chiv_sort1[sort1]

  feh_sort1 = feh[sort1] 
  # need to add the individual error for jj in here as well from the matrix - do later 
  for jj,each,each_id in zip(counts, stars_scaled_sort1,stars_id_sort1):
      each = stars_scaled_sort1[jj] 
      each_err1 = stars_scaled_sort1_errs[jj] 
      each_id = stars_id_sort1[jj] 
      ind_each = list(stars_id_sort1).index(each_id) 
      if ind_each < 150:
         # this is just one star minus the closest 150 
         chival =  ((each - stars_scaled_sort1[0:ind_each+150])**2)**0.5
         # now multiply by the inverse covariance matrix 
         ctake = cinv_sort1[0:150,:,:]
         test = dot(chival, ctake**0.5) 
         test2 = test*test
         test3 =sum(test2**0.5,axis=1)
         # above doing summing to make sense of dimensions 
         chi_ind = stars_id_sort1[0:ind_each+150]
         # then do the same below for the rest 
      if ind_each >= 150:
        ctake = cinv_sort1ind_each-150:ind_each+150,:,:]
        chival =  ((each - stars_scaled_sort1[ind_each-150:ind_each+150]))**2
        chi_ind = stars_id_sort1[ind_each-150:ind_each+150]
        test = dot(chival, ctake) 
        test2 = test*test
        test3 =sum(test2**0.5,axis=1)
      chi2_covs = [sum(a)**0.5 for a in test3]
      chi2_covs = array(chi2_covs) 
      chi2_single_sort1 = argsort(chi2_covs,axis=0) 
      chi2_values_sorted = chi2_covs[chi2_single_sort1]
      chi2_ind_sorted = chi_ind[chi2_single_sort1]
      chi2_all.append(chi2_values_sorted)
      ids_all.append(chi2_ind_sorted) 
  chi2_all = array(chi2_all)
  ids_all = array(ids_all)
  return chi2_all, ids_all 

