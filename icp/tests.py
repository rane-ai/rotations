from icp_experiments import t2v_lie, v2t_lie, v2t_euc, t2v_euc, v2t_q, t2v_q
import numpy as np

'''
Tests for different transformation increment parameterisations
'''

def test_lie():
    v = np.array([0, 0.1, 0.3, 2, 0.5, 0])
    err = np.linalg.norm( t2v_lie( v2t_lie(v) ) - v )
    assert err < 1e-9, err


def test_euc():
    
    T_init = np.asarray([[ 0.862, 0.011, -0.507, 3],
                         [-0.139, 0.967, -0.215, 0.7],
                         [ 0.487, 0.255, 0.835, -2.4], 
                         [ 0.0,   0.0,    0.0,  1.0]])

    T_recon = v2t_euc(t2v_euc(T_init))
    e = np.linalg.norm( T_init.flatten() - T_recon.flatten() )
    assert e < 1e-2

    expected = np.array([0,0,0,0,0,0])
    e = np.linalg.norm( t2v_euc( np.identity(4) ) - expected )
    assert e < 1e-2


def test_q():
    x_vec = np.array([ 0, 0, 0, 0, 0, 0 ])
    e = np.linalg.norm( np.eye(4).flatten() - v2t_q(x_vec).flatten() )
    assert e < 1e-9

    expected = np.array([[-0.2808234, -0.9493278,  0.1411200, 1],
                         [ 0.7828287, -0.3116316, -0.5385768, 2],
                         [ 0.5552634, -0.0407722,  0.8306745, 3],
                         [ 0,          0,          0,         1 ]])

    x_vec = np.array([ 1, 2, 3, 0.2236815, -0.1860895, 0.7783202 ])

    e = np.linalg.norm( expected.flatten() - v2t_q(x_vec).flatten() )
    assert e < 1e-6, e

    e = np.linalg.norm( t2v_q(expected) - x_vec )
    assert e < 1e-7, e


if __name__ == "__main__":
    test_euc()
    test_lie()
    test_q()

