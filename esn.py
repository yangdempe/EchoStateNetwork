"""A simple Python implementation of Echo State Network.

Translated from Matlab scipts on http://www.scholarpedia.org/article/Echo_state_network. 

Author: Zeqian Li

=======================

Jul 20, 2018. I wrote this a while ago. It worked but I'm not sure. Everything is
    pretty messy. 

    I'll just leave it like this. Email me if you have questions. 
"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from sklearn import linear_model


def linreg(S, D):
    """Linear regression algorithm

    S: ntime x (ninput+ninternal)
    D: ntime x noutput
    """
    return (np.linalg.pinv(S) @ D).T


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def identity(x):
    return x


def tanh(x):
    return np.tanh(x)


def arctanh(x):
    return np.arctanh(x)


def freq_gen(length, minperiod, maxperiod):
    """Frequency generator

    Args:
        length: timepoint length
        minperiod: minimum output frequency (inclusive)
        maxperiod: maximum output frequency (exclusive)

    Returns: (fs, sins)
        fs: A length x 1 array of frequencies in range [minf, maxf)
        sins: A length x 1 array of corresponding signal

    """
    inputs = np.ones((2, length))
    outperiod = np.zeros(length)
    rands = np.random.rand(length)
    cur = rands[0]
    for i, rand in enumerate(rands):
        if rand < 0.002:
            cur = np.random.rand()
        outperiod[i] = cur
        inputs[1, i] = -cur + 1
    cur_sin_arg = 0
    outputs = np.zeros((1, length))
    for i in range(1, length):
        currentOutPeriodLength = outperiod[i - 1] * (maxperiod - minperiod) + minperiod
        cur_sin_arg = cur_sin_arg + 2 * np.pi / currentOutPeriodLength
        outputs[0, i] = (np.sin(cur_sin_arg) + 1) / 2

    return inputs, outputs


def simple_scaling(inputs, outputs):
    ninput, noutput = np.shape(inputs)[0], np.shape(outputs)[0]
    m_in, n_in = np.min(inputs[1:, :].flatten()), np.max(inputs[1:, :].flatten())
    input_scaling = np.ones((1, ninput)) * 2 / (n_in - m_in) / (ninput - 1)
    input_scaling[0, 0] = -(n_in + m_in) / (n_in - m_in)

    m_out, n_out = np.min(outputs.flatten()), np.max(outputs.flatten())
    feedback_scaling = 1 / noutput
    output_scaling = 2 / (n_out - m_out)
    output_shift = -(m_out + n_out) / 2

    return {'input_scaling': input_scaling,
            'feedback_scaling': feedback_scaling,
            'output_scaling': output_scaling,
            'output_shift': output_shift}


class ESN:
    """Echo State Network (mostly translated from http://scholarpedia.org/article/Echo_state_network)"""

    def __init__(self, ninput, ninternal, noutput, p=None,
                 spectral_radius=1,
                 input_scaling=1, output_scaling=1, output_shift=0, feedback_scaling=1,
                 activation=tanh, out_activation=identity, invout_activation=identity,
                 dynamics='plain', regression=linreg,
                 noise_level=0,
                 inspect=True):
        """
        Args:
            ninput, ninteral, ntotal: # input/internal/output units
                note that ninput include bias.

        Kwargs:
            spectral_radius: spectral radius
            p: p of the sparse reservoir matrix,
                            if None, self.p = min(10 / ninternal, 1) (critical state)
                            
            input_scaling: scalar; (1 x ninput)
            output_scaling: scalar; (noutput x 1)
            output_shift: scalar; (noutput x 1)        
            feedback_scaling: scalar; (1 x output). Set 0 if no feedback required.
                    W_in *= input_scaling 
                    output_actual = outputs * output_scaling + output_shift
                    W_fb *= feedback_scaling
                    
            activation: reservoir activation function, tanh by default
            out_activation: output activation function, identity by default
            invout_activation: inverse output activation function, identity by default
            dynamics: a string indicating the dynamics in the network. 'leaky' by default
                    'leaky': leaky integrator
            feedback: whether feedback
            noise_level: noise level. A small amount of uniform noise is added when computing interal states
            inspect: show messages

        Other Attributes:
            ntotal: total units
            W: (ninternal x ninternal) \in (-1,1).
            W_in: (ninteral x ninput) \in (-1,1). Scaled by input_scaling.
            W_fb: (ninteral x noutput) \in (-1,1). Scaled by feedback_scaling.
            W_out: (noutput x ninternal) TRAINABLE.

        """
        # check format error
        if np.shape(input_scaling) not in ((), (1, ninput)):
            raise ValueError("input_scaling dimension error")
        if np.shape(output_scaling) not in ((), (noutput, 1)):
            raise ValueError("output_scaling dimension error")
        if np.shape(output_shift) not in ((), (noutput, 1)):
            raise ValueError("output_shift dimension error")
        if np.shape(feedback_scaling) not in ((), (1, noutput)):
            raise ValueError("feedback dimension error")

        # TODO: teacher forcing (different from feedback)
        self.ninput = ninput
        self.ninternal = ninternal
        self.noutput = noutput
        self.ntotal = ninput + ninternal + noutput

        self.spectral_radius = spectral_radius
        self.p = min(10 / self.ninternal, 1) if p is None else p

        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.output_shift = output_shift
        self.feedback_scaling = feedback_scaling

        def init_internal_weights():
            while True:
                try:
                    internal_weights = sp.rand(ninternal, ninternal, density=self.p, format='csr')
                    internal_weights[internal_weights.nonzero()] -= 0.5
                    maxval = max(abs(splin.eigs(internal_weights, k=1, which='LM')[1]))[0]
                    internal_weights = internal_weights / maxval * spectral_radius
                    break
                except splin.ArpackNoConvergence:
                    pass
            return internal_weights

        self.W = init_internal_weights()  # (ninternal x ninternal)
        self.W_in = (2.0 * np.random.rand(ninternal, ninput) - 1.0) * self.input_scaling  # (ninternal x ninput)
        self.W_fb = (2.0 * np.random.rand(ninternal, noutput) - 1.0) * self.feedback_scaling  # (ninternal x ninput)
        self.W_out = np.zeros((noutput, ninternal + ninput))  # noutput x (ninternal + ninput), (-1, 1)

        self.activation = activation
        self.out_activation = out_activation
        self.invout_activation = invout_activation

        dynamics_options = {'leaky': self.leaky, 'plain': self.plain}
        if dynamics in dynamics_options:
            self._update = dynamics_options[dynamics]
        else:
            self._update = dynamics

        self.noise_level = noise_level
        self.inspect = inspect

        self.regression = regression
        self.trained = False
        self._last_input = np.zeros((self.ninput, 1))  # scaled
        self._last_state = np.zeros((self.ninternal, 1))
        self._last_output = np.zeros((self.noutput, 1))  # scaled

        self.leakage = 0.5

    def fit(self, inputs, outputs, nforget=0, inspect=False):
        """Train ESN

        Args/Kwargs:
            inputs: ninput x ntime
            outputs: noutput x ntime
            nforget: first nforget timepoints are discarded in regression
            inspect: output status / show plots for inspection

        """

        # check format
        if len(np.shape(inputs)) != 2 or np.shape(inputs)[0] != self.ninput:
            raise ValueError("ESN fit dimension error: inputs ")
        if len(np.shape(outputs)) != 2 or np.shape(outputs)[0] != self.noutput:
            raise ValueError("ESN fit dimension error: outputs/teachers ")
        if np.shape(inputs)[1] != np.shape(outputs)[1]:
            raise ValueError("ESN fit dimension error: different input/output dimension")

        outputs_scaled = self._scale_outputs(outputs)

        ntime = inputs.shape[1]

        # harvest
        if inspect:
            print("Harvesting states...")
        states = np.zeros((self.ninternal, ntime))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs_scaled[:, t - 1])

        # fit
        if inspect:
            print("fitting...")
        S = np.vstack((states, inputs)).T[nforget:]
        D = self.invout_activation(outputs_scaled.T[nforget:])
        self.W_out = self.regression(S, D)

        # Remember last state?
        self._last_input = inputs[:, -1]
        self._last_state = states[:, -1]
        self._last_output = outputs_scaled[:, -1]

        self.trained = True

        return states

    def predict(self, inputs, turnoff_noise=True, continuing=True):
        """Predict. Can be called only after trained.

        Args:
            inputs: (ninput x ntime)
            continuing: continue from the last state of training
            turnoff_noise: whether turn off noise

        Returns: outputs: noutput x ntime
        """

        # check format
        if len(np.shape(inputs)) != 2 or np.shape(inputs)[0] != self.ninput:
            raise ValueError("ESN predict dimension error: inputs ")
        if not self.trained:
            raise TypeError("ESN not trained.")

        if turnoff_noise:
            self.noise_level = 0
        if not continuing:
            self._last_input = np.zeros((self.ninput, 1))
            self._last_state = np.zeros((self.ninternal, 1))
            self._last_output = np.zeros((self.noutput, 1))

        ntime = inputs.shape[1]
        outputs_scaled = np.zeros((self.noutput, ntime))

        states = np.zeros((self.ninternal, ntime))
        states[:, 0] = self._update(self._last_state, inputs[:, 0], self._last_output)
        outputs_scaled[:, 0] = \
            self.out_activation(self.W_out @ np.hstack((states[:, 0], inputs[:, 0])))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs_scaled[:, t - 1])
            outputs_scaled[:, t] = \
                self.out_activation(self.W_out @ np.hstack((states[:, t], inputs[:, t])))

        outputs = self._unscale_outputs(outputs_scaled)
        return outputs

    def leaky(self, previous_internal, new_input, previous_output):
        """Update ESN by leaky integrator model:
            new_internal = (1 - self.leakage) * previous_internal \
                                + f(W_in @ new_input + W_internal @ previous_internal + W_fb @ previous_feedback)
                                + noise_level * noise

        previous_state: previous internal state at t = n
        new_input: new input at t = n + 1
        previous_output: previous teacher state at t = n

        returns: new internal state
        """

        new_internal = (1 - self.leakage) * previous_internal \
                       + self.activation(self.W_in @ new_input
                                         + self.W @ previous_internal
                                         + self.W_fb @ previous_output) \
                       + self.noise_level * (np.random.rand(self.ninternal) - 0.5)
        return new_internal

    def plain(self, previous_internal, new_input, previous_output):
        """Update ESN by plain model:
            new_internal = f(W_in @ new_input + W_internal @ previous_internal + W_fb @ previous_feedback)
                                + noise_level * noise

        previous_state: previous internal state at t = n
        new_input: new input at t = n + 1
        previous_output: previous teacher state at t = n

        returns: new internal state
        """
        new_internal = self.activation(self.W_in @ new_input
                                       + self.W @ previous_internal
                                       + self.W_fb @ previous_output) \
                       + self.noise_level * (np.random.rand(self.ninternal) - 0.5)
        return new_internal

    def _scale_outputs(self, outputs):
        """scale teacher"""
        return outputs * self.output_scaling + self.output_shift

    def _unscale_outputs(self, outputs_scaled):
        """inverse operation of scaling teacher"""
        return (outputs_scaled - self.output_shift) / self.output_scaling


def mean_square_error(predict, expect):
    return np.sqrt(sum((predict.flatten() - expect.flatten()) * (predict.flatten() - expect.flatten())) / max(
        predict.shape) / np.var(expect))


def wiener_hopf(S, D, alpha=0):
    return (np.linalg.inv(S.T @ S + alpha ** 2 * np.eye(S.shape[1])) @ (S.T @ D)).T
