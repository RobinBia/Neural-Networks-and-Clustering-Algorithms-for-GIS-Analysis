
"""
pfcm.py : Possibilistic Fuzzy C-means clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist
from normalize_columns import normalize_columns, normalize_power_columns
from _fcm import cmeans

def gamma_f(data, centers_fcm, u_fcm, m, k): #data ist x_k und centers_fcm ist v_i
    u = u_fcm ** m

    d = _distance(data, centers_fcm)
    d = np.fmax(d, np.finfo(np.float64).eps)

    gamma = k*(np.divide(np.multiply(u, d ** 2).sum(axis=1), u.sum(axis=1))).T
    #Wenn man entlang der zweiten Axe (axis=1) aufsummiert, erhält man einen Spaltenvektor!!!!
    #gamma muss für später aber zu einem Zeilenvektor gemacht werden
    #In y-Richtung hat man cluster und in x-Richtung die Datenpunkte (u_i_k)
    #In diesem Fall muss der neue Vektor die Dimension i haben.

    return gamma

def _pfcm0(data, centers_old, gamma, c, m, eta, a, b):
    """ Single step of pfcm"""

    dist = _distance(data, centers_old)
    dist = np.fmax(dist, np.finfo(np.float64).eps)#Nullen vorsichtshalber durch minimum Wert ersetzen"


    #Zugehörigkeitsgrade berechnen (u_i_k)
    u = normalize_power_columns(dist, - 2. / (m - 1))

    #Berechnung der Typischkeitswerte
    #Wichtige Funktionen:
    #np.reciprocal: reciprocal(x) entspricht 1/x, nur Elementweise
    #np.ones: Matrix mit Dimension (x,y) mit Einsen gefüllt, x Anzahl der Zeilen, y Spalten
    # Compute typicality values
    d2 = dist ** 2
    d2 = np.fmax(d2, np.finfo(np.float64).eps)
    t = np.reciprocal(np.add(np.ones((c, data.shape[0])), 
                            (np.divide(np.multiply(d2, b).T, gamma).T ** (1. / (eta - 1)))))


    #Clusterzentroiden berechnen (v_i)
    #ut entspricht der Summe (a*u_i_k**m + b*t_i_k**eta)
    ut = np.add(np.multiply((u ** m), a), np.multiply((t ** eta), b))
    centers = np.divide((ut.dot(data)).T, (ut.sum(axis=1)).T).T

    #Berechnung der Zielfunktion J_m_eta
    d_new = _distance(data, centers)
    d_new = np.fmax(d_new, np.finfo(np.float64).eps)
    d_new = d_new ** 2
    d_new = np.fmax(d_new, np.finfo(np.float64).eps)
    jm = np.multiply(ut, d_new).sum() + np.multiply(gamma, (np.subtract(np.ones((c, data.shape[0])), t) ** eta).sum(axis=1).T).sum()


    return centers, u, t, jm


def _distance(data, centers):
    """
    Euclidean distance from each point to each cluster center.
    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.
    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers).T



def pfcm(data, c, m, eta, a, b, k, error_fcm, error_pfcm, maxiter_fcm, maxiter,init_cntr=None, init=None, seed=None):
    """
    Possibilistic fuzzy c-means clustering algorithm
    Parameters
    ----------
    data : 2d array, size (N, S)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    eta : float
        Array exponentiation applied to the typicality function t_old at each
        iteration, where T_new = t_old ** eta.
    a   :   float
        Weighting parameter for the membership function.
    b   : float
        Weighting parameter for the typicality function.
    k   : float
        Weighting parameter for the gamma function.
    error_fcm : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    error_pfcm  : float
        Stopping criterion for pfcm.
    maxiter_fcm : int
        Maximum number of iterations allowed in FCM.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    init_ctr: 2d array, sizy (c, S)
        Initial cluster prototypes. If none provided, algorithm is randomly
        initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    Returns
    -------
    cntr : 2d array, size (c, S)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (c, N)
        Final fuzzy c-partitioned matrix.
    t : 2d array, (c, N)
        Final typicality matrix.
    d : 2d array, (c, N)
        Final Euclidian distance matrix.
    jmeta : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
        :type init_cntr: object

    """

    # Start FCM algorithm
    if init_cntr is None:
        centers_fcm, u_fcm, u0_fcm, d_fcm, jm_fcm, number_of_iter_fcm, fpc_fcm = cmeans(data.T, c, m, error_fcm,
                                                                                        maxiter_fcm, init=None)
    else:
        centers_fcm, u_fcm, u0_fcm, d_fcm, jm_fcm, number_of_iter_fcm, fpc_fcm = cmeans(data.T, c, m, error_fcm,
                                                                                        maxiter_fcm, init=None,
                                                                                        init_cntr=init_cntr.T)

    # Compute the gamma function
    gamma = gamma_f(data, centers_fcm, u_fcm, m, k)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    centers = (centers_fcm).copy()
    # Main pfcm loop
    while p < maxiter - 1:
        centers2 = centers.copy() 
        [centers, u, t, Jjm] = _pfcm0(data, centers2, gamma, c, m, eta, a, b) #Schritt 1+2
        jm = np.hstack((jm, Jjm)) #In jedem Durchlauf wird an ein Array jm die neue Zielfunktion Jjm angehängt.
        p += 1

        # Stopping rule
        if np.linalg.norm(centers - centers2) < error_pfcm:
            break

    return centers, u, t, jm, p