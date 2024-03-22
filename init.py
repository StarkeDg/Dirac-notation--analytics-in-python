import numpy as np
import sympy as sym
from sympy.physics.quantum import TensorProduct as tp
from sympy.physics.quantum import Ket, Bra
import math
from functools import reduce
import copy
import re


def mysim(expr):
    """
    Do jeito que tá não funciona. Ainda precisa importar simbolos pra usar.
    """
    return sym.simplify(expr, rational=True)


'''
Auxiliary functions
'''


def inner_product(v, w):
    return np.inner(np.conjugate(v), w)


def norm(v):
    return np.linalg.norm(v)


def cb(n, j):
    """
    Args:
        n (int): levels
        j (int): element that will be non-zero

    Returns:
        array: returns a standard base vector of C^n
    """
    vec = sym.zeros(n, 1)
    vec[j] = 1
    return vec


def proj(psi):
    '''returns the projector in the psi vector'''
    d = psi.shape[0]
    P = sym.zeros(d, d)
    for j in range(0, d):
        for k in range(0, d):
            P[j, k] = psi[j] * np.conjugate(psi[k])
    return P


def tsco(A):
    '''returns the conjugate transpose'''
    return sym.conjugate(sym.transpose(A))


'''
Generic vector state
'''


def psi_c(n, word='c', mtype=0):
    """
    Args:
        n (int): state vector levels
        mtype (int): array type, column array or row array. Zero (0) for
            row matrix and one (1) for row matrix

    Returns:
        array: Returns a row or column array depending on the
               type (tp) selected with n levels.
    """
    if mtype == 1:
        return sym.Matrix(n, 1, lambda i, j: f'{word}{i}').T
    else:
        return sym.Matrix(n, 1, lambda i, j: f'{word}{i}')


"""
Same situation as the function above, but write the coefficients 
with indices in binary form. Useful to verify some patterns after 
the actions of certain ports like CNOT among other utilities.
"""


def psi_bin(n, word='c', mtype=0):
    if mtype == 1:
        return sym.Matrix(n, 1, lambda i, j: "c_" + "{:b}".format(int(i)).zfill(math.ceil(sym.log(n, 2)))).T
    else:
        return sym.Matrix(n, 1, lambda i, j: "c_" + "{:b}".format(int(i)).zfill(math.ceil(sym.log(n, 2))))


"""
Interesting function to say exactly which matrix values
are different from zero and at the same time have a visualization
of the state vector as if it were written in Dirac notation
"""


def plus(d):
    return (1 / sym.sqrt(2)) * (cb(d, 0) + cb(d, 1))


def minus(d):
    return (1 / sym.sqrt(2)) * (cb(d, 0) - cb(d, 1))


def o_plus(d):
    # noinspection PyTypeChecker
    return (1 / sym.sqrt(2)) * (cb(d, 0) + 1j * cb(d, 1))


def o_minus(d):
    # noinspection PyTypeChecker
    return (1 / sym.sqrt(2)) * (cb(d, 0) - 1j * cb(d, 1))


def phi_plus(d):
    return (1 / sym.sqrt(2)) * (tp(cb(d, 0), cb(d, 0)) + tp(cb(d, 1), cb(d, 1)))


def phi_minus(d):
    return (1 / sym.sqrt(2)) * (tp(cb(d, 0), cb(d, 0)) - tp(cb(d, 1), cb(d, 1)))


def psi_plus(d):
    return (1 / sym.sqrt(2)) * (tp(cb(d, 0), cb(d, 1)) + tp(cb(d, 1), cb(d, 0)))


def psi_minus(d):
    return (1 / sym.sqrt(2)) * (tp(cb(d, 0), cb(d, 1)) - tp(cb(d, 1), cb(d, 0)))


def pbk(seq, dim=2, mtype=0):
    """
    Args:
        seq (str): input sequence. Ex: '010110'
        dim (int): computational basis dimension (default=2 for qubits)
        mtype (int): array type, column array or row array. Zero (default=0) for
            row matrix and one (1) for row matrix

    Returns:
        array: ket vector for the input sequence in the computational basis of dimension 'dim'
    """

    states = {
        "+": plus,
        "-": minus,
        "r": o_plus,
        "l": o_minus,
        "P": phi_plus,
        "Q": phi_minus,
        "R": psi_plus,
        "S": psi_minus
    }

    matrices = [states[state](dim) if state in states else cb(dim, int(state)) for state in seq]
    if mtype == 1:
        return tp(*matrices).T
    return tp(*matrices)


'''
Generic density matrix (
ρ
)
'''

"""
Returns a symbolic generic rho
n = column dimension
Returns a symbolic generic rho
square matrix
n = row/column levels
"""

def rho_g(n, word='0'):
    if word == '0':
        elementos = [sym.symbols('rho_{:02d}'.format(i + 1)) for i in range(n ** 2)]
        return sym.Matrix(n, n, elementos)
    else:
        elementos = [sym.symbols('{}_{:02d}'.format(word, i + 1)) for i in range(n ** 2)]
        return sym.Matrix(n, n, elementos)


"""
Returns a symbolic generic rho with indices in binary form
square matrix
n = row/column levels
"""

def rho_bin(n, word='0'):
    if word == '0':
        elementos = ['rho_' + bin(i + 1)[2:].zfill(int(math.ceil(sym.log(n ** 2, 2)))) for i in range(n ** 2)]
        return sym.Matrix(n, n, elementos)
    else:
        elementos = [f'{word}_' + bin(i + 1)[2:].zfill(int(math.ceil(sym.log(n ** 2, 2)))) for i in range(n ** 2)]
        return sym.Matrix(n, n, elementos)


#####################################################################################
'''
Dirac notation
'''

"""
Return Psi or rho in bra-ket notation
Sent Psi - return Psi in bra-ket notation
Send rho - return rho in bra-ket notation
"""


def mbk(matrix, dim=2):
    """
    Args:
        matrix (sym.matrices.dense.MutableDenseMatrix): The input must be a sympy array.
                                            It can be a state vector or a density matrix
    Returns:
        Prints the matrix in Dirac notation
    """

    def convert_dim(x, dim, min_digits):
        """
        Helper function that checks how many digits the bra-ket will
        have based on the dimension and size of the input matrix
        """
        digits = '0123456789'[:dim]
        result = ''
        while x > 0 or len(result) < min_digits:
            x, digit = divmod(x, dim)
            result = digits[digit] + result
        return result

    pos_bin = []
    val = []
    if isinstance(matrix, sym.matrices.dense.MutableDenseMatrix):
        n_linhas, n_colunas = matrix.shape
        """
        If the row == 1 the notation will be Bra, if the column == 1 the notation will be Ket and if 
        it doesn't fit in any of these cases, then it's because it's a density matrix and will be handled by 'else'
        """
        if n_linhas == 1:
            # bra
            Psi = 0
            x = len(matrix)
            for i in range(x):
                if matrix[i] != 0:
                    val.append(matrix[i])
                    pos.append(i)
                    pos_bin.append(convert_dim(i, dim, int(math.ceil(math.log(x) / math.log(dim)))))
            for i in range(len(val)):
                Psi = val[i] * Bra(pos_bin[i]) + Psi
            return sym.simplify(Psi)
        if n_colunas == 1:
            # ket
            Psi = 0
            x = len(matrix)
            for i in range(x):
                if matrix[i] != 0:
                    val.append(matrix[i])
                    pos_bin.append(convert_dim(i, dim, int(math.ceil(math.log(x) / math.log(dim)))))
            for i in range(len(val)):
                Psi += val[i] * Ket(pos_bin[i])
            return Psi
        else:
            # projetor
            m, n = matrix.shape
            rho = 0
            for i in range(m):
                for j in range(n):
                    if matrix[i, j] != 0:
                        val.append(matrix[i, j])
                        pos_bin.append((convert_dim(i, dim, int(math.ceil(math.log(m) / math.log(dim)))), \
                                        convert_dim(j, dim, int(math.ceil(math.log(m) / math.log(dim))))))
            for i in range(len(val)):
                rho = val[i] * (Ket(pos_bin[i][0])) * (Bra(pos_bin[i][1])) + rho
            return rho


########################################################################################

'''
Composite systems

'''


def comp_sys(*psi):
    """
    *psi (sym.matrices.dense.MutableDenseMatrix): The input must be 'n' subsystems as sympy array.
                                                          Must be a state vectors or list of state vector.
                                                          Se examples for more details.
    Returns:
        Returns the tensor product of the state vectors in the order they are entered into the function.
    """
    if (psi).__class__ == tuple and len(psi) == 1:
        order = []
        for j in range(len(*psi)):
            order.append(psi[0][j])
        return sym.Matrix(reduce(tp, [x for x in order]))
    else:
        return sym.Matrix(reduce(tp, [x for x in psi]))


'''
Changing the position of the qubit: changeSS(psi, pos_i, pos_f)

'''


def changeSS(psi, pos_i, pos_f):
    """
    Changes the subsystem from a given position (pos_i) to another position (pos_f).
    Args:
        matrix (sym.matrices.dense.MutableDenseMatrix): The input must be a sympy array.
                                                          Must be a state vector.
        pos_i (int): n-1, n-2,..., 1, 0.
        pos_f (int): n-1, n-2,..., 1, 0.
            where n is the number of subsystems.
    Returns:
        Returns the state vector with the qubit in the desired position.
    """

    def move_bit(binary, pos_i, pos_f, n):
        characters = list(binary)
        if n - 1 - pos_i == n - 1 - pos_f:
            return binary
        bit = characters.pop(n - 1 - pos_i)
        characters.insert(n - 1 - pos_f, bit)
        binm = ''.join(characters)
        return binm

    lenp = len(psi)
    n = math.ceil(math.log(lenp, 2))
    psi_f = [0] * lenp
    for j in range(lenp):
        psi_f[int(move_bit('{:0{}b}'.format(j, n), pos_i, pos_f, n), 2)] = psi[j]
    return sym.Matrix(psi_f)


'''
Tensor products for quantum logic gates: tp_gate(n, gate, ssystem)

'''


def tp_gate(n, gate, *ssystem):
    """
    Adjusts the matrix for the subsystems that will act the quantum logic gate (QLG).
    Args:
        n (int): is the number of the subsystems
        gate (sym.matrices.dense.MutableDenseMatrix): 2x2 dimension QLG matrix
        ssystem (int): subsystems that QLG will act (subsystems position in the bra-ket).
                        0 ≤ ssystem ≤ n-1
    Returns:
        Matrix adjusted to act the QLG in a certain subsystem considering that we have n subsystems
    """
    if len(ssystem) == 1 and isinstance(ssystem[0], (list, tuple)):
        lst = list(ssystem[0])
    else:
        lst = list(ssystem)
    lst.sort(reverse=False)
    order = []
    for j in range(n - 1, -1, -1):
        if j in lst:
            order.append(gate)
        else:
            order.append(sym.eye(2))
    return tp(*order)


'''
Acting a quantum logic gate in a state: gatep(gate, psi, ssystem)

'''


def gatep(gate, psi, ssystem):
    n = int(np.log(len(psi), 2))
    if n == 1:
        return gate * psi
    else:
        tpgate = tp_gate(n, gate, ssystem)
        return tpgate * psi


'''
Tensor products for controlled quantum logic gates:

'''


def tp_ctrl(n, gate, ctrl_act, target):
    """
    Adjusts the matrix for the subsystems that will act the controlled quantum logic gate (CQLG).

    Args:
        n (int): é a quantidade de subsistemas envolvidos
        gate (sym.matrices.dense.MutableDenseMatrix): 2x2 2ension quantum logic gate array that you want
                                                        to be controlled by other subsystems
        *ctrl (list): is the controls of CQLG - 0 ≤ ctrl ≤ n-1. Example with multi controls: (2,1), (1,3)
        target (int): is the target of CQLG - 0 ≤ target ≤ n-1
                        target nedds to be different of ctrl
        act (int): CQLG enable bit.
                    1: 1 to be the enable bit (default)
                    0: 0 to be the enable bit

    Returns:
        Matrix adjusted to act the QLG in a certain subsystem considering that we have n subsystems
    """
    order = []
    if isinstance(ctrl_act, tuple):
        ctrl_act = list(ctrl_act)
    elif isinstance(ctrl_act, tuple) and isinstance(ctrl_act[0], tuple):
        ctrl_act = [list(tupla) for tupla in ctrl_act]
    if isinstance(target, tuple):
        target = list(target)
    elif isinstance(target, int):
        target = [target]
    if len(ctrl_act) >= 2 and isinstance(ctrl_act[1], int):
        ctrl_lst = []
        if isinstance(ctrl_act[0], int):
            ctrl_lst = ctrl_act[0:-1]
            act = [ctrl_act[-1]] * len(ctrl_lst)
        else:
            act = [ctrl_act[1]] * len(ctrl_act[0])
            ctrl_lst = list(ctrl_act[0])
    else:
        ctrl_lst, act = [], []
        for el in ctrl_act:
            ctrl_lst.append(el[0])
            act.append(el[1])
    #
    # Verifications
    #
    for el in target:
        if el in ctrl_lst:
            return print('Target qubit cannot be control either')
    if len(target) + len(ctrl_lst) > n:
        return print('There are more targets and controls than subsystems')
    for val in ctrl_lst:
        if val >= n or val < 0:
            return print('The control is outside the allowed limits')
    for val in target:
        if val >= n or val < 0:
            return print('The target is outside the allowed limits')
    #
    #
    #
    for i in range(n - 1, -1, -1):
        if i in target:
            order.append((gate, sym.eye(2)))
        elif i in ctrl_lst:
            if act[ctrl_lst.index(i)] == 0:
                order.append((proj(cb(2, 0)), proj(cb(2, 1))))
            if act[ctrl_lst.index(i)] == 1:
                order.append((proj(cb(2, 1)), proj(cb(2, 0))))
        else:
            order.append((sym.eye(2), sym.eye(2)))
    CGate = sym.Matrix(reduce(tp, [x[0] for x in order]) + reduce(tp, [x[1] for x in order]))
    for j in range(CGate.rows):
        flag = 0
        for k in range(CGate.cols):
            if CGate[j, k] != 0:
                flag = 1
                break
        if flag == 0:
            CGate[j, j] = 1
    return CGate


'''
Acting a controlled quantum logic gate in a state: ctrlp(gate, psi, ctrl_act, target)

'''


def ctrlp(gate, psi, ctrl_act, target):
    """
    Args:
        matrix (sym.matrices.dense.MutableDenseMatrix): The input must be a sympy array.
                                                          Must be a state vector.
        ctrl (int): is the control of CQLG - 0 ≤ ctrl ≤ n-1
        target (int): is the target of CQLG - 0 ≤ target ≤ n-1
        act (int): CQLG enable bit.
                    1: 1 to be the enable bit (default)
                    0: 0 to be the enable bit

    Returns:
        Returns the state vector evolved by CQLG
    """
    n = int(np.log(len(psi), 2))
    ctrl_gate = tp_ctrl(n, gate, ctrl_act, target)
    return ctrl_gate * psi


'''
Z-, X-, Y- and Bell's basis
'''


def mbk_xyBB(matrix, *bpos, split=-1):
    #############################################################################
    def split_string(string):
        pattern = r'(PP|QQ|RR|SS|[ab]+|[cd]+|[01]+)'
        substrings = re.split(pattern, string)
        substrings = [substring for substring in substrings if substring]
        return substrings

    #############################################################################
    base = [t[0] for t in bpos]
    lst = [t[1:] for t in bpos]
    matrix_aux = copy.deepcopy(matrix)
    for j in range(len(base)):
        if base[j] == 'x' or base[j] == 'y':
            for k in range(len(lst[j])):
                matrix_aux = xyBasis(matrix_aux, lst[j][k], base=base[j])
        elif base[j] == 'BB':
            for k in range(len(lst[j])):
                matrix_aux = BBasis(matrix_aux, lst[j][k])
    val, poss_bin, order = [], [], [[]]
    if isinstance(matrix_aux, sym.matrices.dense.MutableDenseMatrix):
        n_linhas, n_colunas = matrix_aux.shape
        if n_colunas == 1:
            Psi = 0
            x = len(matrix_aux)
            n = int(math.log(x) / math.log(2))
            for i in range(x):
                if matrix_aux[i] != 0:
                    val.append(matrix_aux[i])
                    poss_bin.append(format(i, f'0{int(math.log(x) / math.log(2))}b'))
            for i in range(len(base)):
                for j in range(len(val)):
                    order = []
                    for k in range(len(lst[i])):
                        if base[i] == 'x':
                            if poss_bin[j][n - 1 - lst[i][k]] == str('0'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k]] + 'a' + poss_bin[j][n - lst[i][k]:])
                            elif poss_bin[j][n - 1 - lst[i][k]] == str('1'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k]] + 'b' + poss_bin[j][n - lst[i][k]:])
                        if base[i] == 'y':
                            if poss_bin[j][n - 1 - lst[i][k]] == str('0'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k]] + 'c' + poss_bin[j][n - lst[i][k]:])
                            elif poss_bin[j][n - 1 - lst[i][k]] == str('1'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k]] + 'd' + poss_bin[j][n - lst[i][k]:])
                        if base[i] == 'BB':
                            if poss_bin[j][n - 1 - lst[i][k][0]] == str('0') and poss_bin[j][
                                n - 1 - lst[i][k][1]] == str('0'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k][0]] + 'PP' + poss_bin[j][n - lst[i][k][1]:])
                            if poss_bin[j][n - 1 - lst[i][k][0]] == str('1') and poss_bin[j][
                                n - 1 - lst[i][k][1]] == str('1'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k][0]] + 'QQ' + poss_bin[j][n - lst[i][k][1]:])
                            if poss_bin[j][n - 1 - lst[i][k][0]] == str('0') and poss_bin[j][
                                n - 1 - lst[i][k][1]] == str('1'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k][0]] + 'RR' + poss_bin[j][n - lst[i][k][1]:])
                            if poss_bin[j][n - 1 - lst[i][k][0]] == str('1') and poss_bin[j][
                                n - 1 - lst[i][k][1]] == str('0'):
                                order.append(poss_bin[j][:n - 1 - lst[i][k][0]] + 'SS' + poss_bin[j][n - lst[i][k][1]:])
                        if k < len(lst):
                            poss_bin.pop(j)
                            poss_bin.insert(j, order[k])
                    if i == len(base) - 1:
                        if split == 1:
                            string = order[-1]
                            string = split_string(string)
                            t_substrings = []
                            for substring in string:
                                t_substring = substring.replace("a", "+").replace("b", "-") \
                                    .replace("c", "\oplus").replace("d", "\ominus") \
                                    .replace("PP", "\Phi_{+}").replace("QQ", "\Phi_{-}") \
                                    .replace("RR", "\Psi_{+}").replace("SS", "\Psi_{-}")
                                t_substrings.append(t_substring)
                            Psi += val[j] * sym.Mul(*[Ket(x) for x in t_substrings])
                        else:
                            string = order[-1]
                            string = string.replace("a", "+").replace("b", "-") \
                                .replace("c", "\oplus").replace("d", "\ominus") \
                                .replace("PP", "\Phi_{+}").replace("QQ", "\Phi_{-}") \
                                .replace("RR", "\Psi_{+}").replace("SS", "\Psi_{-}")
                            Psi += val[j] * Ket(string)
            return Psi


def xyBasis(matrix, *pos, base='x'):
    lst = list(pos)
    x = len(matrix)
    n = int(np.log(x, 2))
    psi_mod = copy.deepcopy(matrix)
    psi_f = sym.Matrix([0] * 2 ** n)
    for j in range(x):
        char = list('{:0{}b}'.format(j, math.ceil(math.log(x, 2))))
        if base == 'x':
            plus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0]]) + '0' + ''.join(char[n - lst[0]:])) \
                                       + pbk(''.join(char[:n - 1 - lst[0]]) + '1' + ''.join(char[n - lst[0]:])))
            minus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0]]) + '0' + ''.join(char[n - lst[0]:])) \
                                        - pbk(''.join(char[:n - 1 - lst[0]]) + '1' + ''.join(char[n - lst[0]:])))
        if base == 'y':
            plus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0]]) + '0' + ''.join(char[n - lst[0]:])) \
                                       + pbk(''.join(char[:n - 1 - lst[0]]) + '1' + ''.join(char[n - lst[0]:])))
            minus = (-1j / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0]]) + '0' + ''.join(char[n - lst[0]:])) \
                                          - pbk(''.join(char[:n - 1 - lst[0]]) + '1' + ''.join(char[n - lst[0]:])))
        if char[n - 1 - lst[0]] == str(0):
            # print('1st')
            psi_f = psi_mod[j] * plus + psi_f
            psi_mod[j] = 0
        if char[n - 1 - lst[0]] == str(1):
            # print('2nd')
            psi_f = psi_mod[j] * minus + psi_f
            psi_mod[j] = 0
    psi_f = psi_f + psi_mod
    return psi_f


def BBasis(matrix, *pos):
    lst = list(pos)
    x = len(matrix)
    n = int(np.log(x, 2))
    psi_mod = copy.deepcopy(matrix)
    psi_f = sym.Matrix([0] * 2 ** n)
    for j in range(x):
        char = list('{:0{}b}'.format(j, math.ceil(math.log(x, 2))))
        # 00 = phi_plus, 01 = phi_minus, 10 = psi_plus, 11 = psi_minus
        phi_plus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0][0]]) + '00' + ''.join(char[n - lst[0][1]:])) \
                                       + pbk(''.join(char[:n - 1 - lst[0][0]]) + '11' + ''.join(char[n - lst[0][1]:])))
        phi_minus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0][0]]) + '00' + ''.join(char[n - lst[0][1]:])) \
                                        - pbk(''.join(char[:n - 1 - lst[0][0]]) + '11' + ''.join(char[n - lst[0][1]:])))
        psi_plus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0][0]]) + '01' + ''.join(char[n - lst[0][1]:])) \
                                       + pbk(''.join(char[:n - 1 - lst[0][0]]) + '10' + ''.join(char[n - lst[0][1]:])))
        psi_minus = (1 / np.sqrt(2)) * (pbk(''.join(char[:n - 1 - lst[0][0]]) + '01' + ''.join(char[n - lst[0][1]:])) \
                                        - pbk(''.join(char[:n - 1 - lst[0][0]]) + '10' + ''.join(char[n - lst[0][1]:])))
        if char[n - 1 - lst[0][0]] == str(0) and char[n - 1 - lst[0][1]] == str(0):
            psi_f = psi_mod[j] * phi_plus + psi_f
            psi_mod[j] = 0
        if char[n - 1 - lst[0][0]] == str(1) and char[n - 1 - lst[0][1]] == str(1):
            psi_f = psi_mod[j] * phi_minus + psi_f
            psi_mod[j] = 0
        if char[n - 1 - lst[0][0]] == str(0) and char[n - 1 - lst[0][1]] == str(1):
            psi_f = psi_mod[j] * psi_plus + psi_f
            psi_mod[j] = 0
        if char[n - 1 - lst[0][0]] == str(1) and char[n - 1 - lst[0][1]] == str(0):
            psi_f = psi_mod[j] * psi_minus + psi_f
            psi_mod[j] = 0
    psi_f = psi_f + psi_mod
    return psi_f
