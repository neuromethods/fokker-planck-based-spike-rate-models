from __future__ import print_function
from brian2 import *
import time
import os
import shutil
from utils_net import smooth_trace, fixed_connectivity





cpp_default_dir = 'brian2_compile'


def network_sim(signal, params, rec = False, standalone_dir = cpp_default_dir):

    # todo only one network simulation for a,b = const. and a(t),b(t)
    # combine PIF, EIF, LIF in one simulation code

    if params['brian2_standalone']:
        # build on run = False (changed for brian2_rc3)
        set_device(params['brian2_device'], build_on_run=False)
        device.insert_code('main', 'srand('+str(int(time.time())+os.getpid())+');')
        standalonedir = 'standalone_{}_pid{}'.format(time.strftime("%Y-=%m-%dT%H:%M:%S"), os.getpid())
    else:
        prefs.codegen.target = 'numpy'

    #stripe on brian units within the copied dict for the simulation so that brian can work with them
    dt_sim =        params['net_dt']*ms
    C =             params['C']*pF
    gL =            params['gL']*nS
    taum =          params['taum']*ms
    EL =            params['EL']*mV
    Ew =            params['Ew']*mV
    VT =            params['VT']*mV
    negVT = -VT
    deltaT =        params['deltaT']*mV
    Vcut =          params['Vcut']*mV
    tauw =          params['tauw']*ms
    Vr =            params['Vr']*mV
    t_ref =         params['t_ref']*ms
    net_w_init =  params['net_w_init']*pA
    rates_dt = params['net_record_dt'] *ms
    runtime = params['runtime']*ms
    time_r = np.arange(0., runtime/ms, rates_dt/ms)
    time_v = np.arange(0., runtime/ms, dt_sim/ms)
    #ratio used for smoothing the spikehistogram to match resolution
    ratesdt_dtsim=int(rates_dt/dt_sim)
    N_total = params['N_total']
    print(N_total)
    # garbage collect so we can run multiple brian2 runs
    gc.collect()


    # seed our random number generator!  necessary on the unix server
    np.random.seed()

    mu_ext_array = signal[0] # [mV/ms]
    sigma_ext_array = signal[1] # [mV/sqrt(ms)]



    # what is recorded during the simulation
    record_spikes = params['net_record_spikes']
    record_w = params['net_record_w']
    record_v_example_trace = params['net_record_example_v_traces']
    record_all_v_at_times = True if params['net_record_all_neurons'] else False
    record_v_stats = params['net_record_v_stats']
    record_w_stats = params['net_record_w_stats']

    # simulation timestep
    simclock = Clock(dt_sim)

    w_refr = ' (unless refractory)' if 'net_w_refr' not in params or params['net_w_refr'] else ''

    a = params['a']
    b = params['b']
    # convert to array if adapt params are scalar values
    if type(a) in [int,float]:
        a = np.ones_like(mu_ext_array)*a
    if type(b) in [int,float]:
        b = np.ones_like(mu_ext_array)*b

    # decide if there's adaptation
    have_adap = True if  (a.any() > 0.0)  or (b.any() > 0.0) else False
    print(have_adap)

    # convert numpy arrays to TimedArrays
    a = TimedArray(a*nS, dt_sim)
    b = TimedArray(b*pA, dt_sim)





    #transform the external input into TimedArray
    mu_ext = TimedArray(mu_ext_array*(mV/ms), dt_sim)
    sigma_ext = TimedArray(sigma_ext_array*(mV/sqrt(ms)), dt_sim)

    #get the model specific term EIF/PIF
    if params['neuron_model'] == 'EIF' :
        model_term = '((EL - v) + deltaT * exp((negVT + v) / deltaT)) / taum'
    elif params['neuron_model'] == 'PIF':
        model_term = ''
    elif params['neuron_model'] == 'LIF':
        model_term = '(EL - v) / taum'
    else:
        mes = 'The model "{}" has not been implemented yet. For options see params dict.'.format(params['neuron_model'])
        raise NotImplementedError(mes)

    model_eqs = '''
        dv/dt = %s %s + mu_ext(t) + sigma_ext(t) * xi  : volt (unless refractory)
        %s
        ''' % (model_term,'- (w / C)' if have_adap else '', ('dw/dt = (a(t) * (v - Ew) - w) / tauw : amp %s' % w_refr)
               if have_adap else '')

    # initialize Neuron group
    G = NeuronGroup(N = N_total, model = model_eqs,
                    threshold = 'v > Vcut',clock = simclock,
                    reset = 'v = Vr%s' % ('; w += b(t)' if have_adap else ''),
                    refractory = t_ref, method = params['net_integration_method'])


    # initialize PopulationRateMonitor
    rate_monitor = PopulationRateMonitor(G, name = 'aeif_ratemon')

    # intitialize net
    Net = Network(G, rate_monitor)

    if rec:
        print('building synapses...')
        start_synapses = time.time()
        J = params['J']*mV
        K = params['K']
        #synapses object
        #synapses from G --> G! only one population
        #this only specifies the dynamics of the synapses. they get acctually created when the .connect method is called
        Syn = Synapses(G,G,on_pre= 'v+=J')
        sparsity = float(K)/N_total
        assert 0 <= sparsity <= 1.0
        # connectivity type
        if params['connectivity_type'] == 'binomial':
            Syn.connect(True, p = sparsity)
        elif params['connectivity_type'] == 'fixed':
            prelist, postlist = fixed_connectivity(N_total, K)
            Syn.connect(i=prelist, j=postlist)
        # delays
        # no delay; nothing has to be implemented
        if params['delay_type'] == 0:
            pass
        # constant delay
        elif params['delay_type'] == 1:
            Syn.delay = '{} * ms'.format(params['const_delay'])
        # exponentially distributed delays
        elif params['delay_type'] == 2:
            #taud is the mean=standard deviation of the distribution
            Syn.delay = '-log(rand()) * {} * ms'.format(params['taud'])
        # exp. delay dist. + const. delay
        elif params['delay_type'] == 3:
            Syn.delay = ('({} -log(rand()) * {}) * ms'.format(format(params['const_delay']), params['taud']))
        # add synapses to the network
        else:
            raise NotImplementedError
        Net.add(Syn)
        print('build synapses time: {}s'.format(time.time()-start_synapses))

    #initial distribution of the network simulation
    if params['net_v_init'] == 'delta':
        G.v = np.ones(len(G)) * params['net_delta_peak']*mV
    elif params['net_v_init'] == 'normal':
        G.v = params['net_normal_sigma'] * np.random.randn((len(G))) * mV + params['net_normal_mean'] * mV
    elif params['net_v_init'] == 'uniform':
        len_interval = Vcut - Vr
        G.v = np.random.rand(len(G)) * len_interval + Vr

    # initial distribution of w_mean
    if have_adap:
        # standart deviation of w_mean is set to 0.1
        G.w = 0.1 * np.random.randn(len(G)) * pA + net_w_init

    # include a lower bound for the membrane voltage
    if 'net_v_lower_bound' in params and params['net_v_lower_bound'] is not None:
        #new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound = G.run_regularly('v = clip(v, %s * mV, 10000 * mV)'
                                       % float(params['net_v_lower_bound']),
                                when = 'end', order = -1, dt = dt_sim)
        print('Lower bound active at {}'.format(params['net_v_lower_bound']))
        Net.add(V_lowerbound)

    if record_v_example_trace:
        # record v from params['net_record_v_trace'] neurons
        v_monitor_example_trace = StateMonitor(G, 'v',
                                 record = range(min(params['net_record_example_v_traces']
                                                    ,params['N_total'])))
        Net.add(v_monitor_example_trace)


    if record_all_v_at_times:
        # define Clock which runs on a very course time grid (memory issue)
        clock_record_all = Clock(params['net_record_all_neurons_dt']*ms)
        v_monitor_record_all = StateMonitor(G, 'v', record=True, clock = clock_record_all)
        Net.add(v_monitor_record_all)

    if record_v_stats:
        # create statistics neuton 'group' with 1 neuron
        eqs_Gvstats = '''
                        v_mean : volt
                        v_var  : volt**2
                    '''
        Gvstats = NeuronGroup(1, eqs_Gvstats, clock = simclock)
        # connect the two neuron groups with Synapses
        eqs_Sstats = '''
                        v_mean_post = v_pre/N_pre             : volt    (summed)
                        v_var_post  = (v_pre-v_mean)**2/N_pre : volt**2 (summed)
                     '''
        Svstats = Synapses(G, Gvstats, eqs_Sstats)
        Svstats.connect(True)
        VstatsMon = StateMonitor(Gvstats, ['v_mean', 'v_var'], record = True)
        # maybe add this a bit later ... together with all the other stuff
        Net.add(Svstats)
        Net.add(Gvstats)
        Net.add(VstatsMon)

    if record_w_stats and have_adap:
        eqs_Gwstats = '''
                      w_mean : amp
                      w_var  : amp**2
                      '''
        Gwstats = NeuronGroup(1, eqs_Gwstats, clock = simclock)
        eqs_Stats = '''
                    w_mean_post = w_pre/N_pre               : amp (summed)
                    w_var_post = (w_pre-w_mean)**2/N_pre    : amp**2 (summed)
                    '''
        Swstats = Synapses(G, Gwstats, eqs_Stats)
        Swstats.connect(True)
        WstatsMon = StateMonitor(Gwstats, ['w_mean', 'w_var'], record=True)
        Net.add(Swstats)
        Net.add(Gwstats)
        Net.add(WstatsMon)


    if record_spikes > 0:
        record_spikes_group = Subgroup(G, 0, min(record_spikes, N_total))
        spike_monitor = SpikeMonitor(record_spikes_group, name = 'aeif_spikemon')
        Net.add(spike_monitor, record_spikes_group)

    if record_w > 0 and have_adap:
        record_w_group = Subgroup(G, 0, min(record_w, N_total))
        w_monitor = StateMonitor(record_w_group, 'w', record = range(params['net_record_w']),
                                 dt = dt_sim, name = 'aeif_wmon')
        Net.add(w_monitor, record_w_group)

    print('------------------ running network!')
    start_time = time.time()
    Net.run(runtime, report = 'text')

    if params['brian2_standalone']:
        project_dir = cpp_default_dir + '/test' + str(os.getpid())
        device.build(directory = project_dir, compile=True, run=True)



    # extract results

    # function for binnung the population rate trace
    def binning(arr, N):
        if int(N) in [0, 1]:
            return arr
        else:
            len_return = int(len(arr)/N)
            return np.array([np.mean(arr[k:min(k+N,len(arr))]) for k in range(0, len(arr), N)])

    # unbinned quantites
    net_rates = rate_monitor.rate/Hz
    net_t = rate_monitor.t/ms




    if record_v_example_trace > 0:
        v_neurons = v_monitor_example_trace.v/mV
        t_v = v_monitor_example_trace.t/ms

    # old way of saving wm
    if record_w > 0 and have_adap:
        net_w = w_monitor.w / pA
        net_wt = w_monitor.t/ms
    if record_spikes > 0:
        # multiply by 1 like this to ensure brian extracts the results before we delete the compile directory
        net_spikes = spike_monitor.it
        i, t = net_spikes
        i = i * 1; t = t * 1
        net_spikes = [i, t]

    if record_v_stats:
        v_mean = VstatsMon.v_mean[0]/mV
        v_var = VstatsMon.v_var[0]/mV**2
        v_std = np.sqrt(v_var)

    if record_w_stats and have_adap:
         w_mean = WstatsMon.w_mean[0]/pA
         w_var = WstatsMon.w_var[0]/pA**2
         w_std = np.sqrt(w_var)

    if record_all_v_at_times:
        v_all_neurons = v_monitor_record_all.v/mV
        t_all_neurons = v_monitor_record_all.t/ms



    run_time = time.time() - start_time
    print('runtime: %1.1f' % run_time)


    if params['brian2_standalone']:
        shutil.rmtree(project_dir)
        device.reinit()

    #for smoothing function net_rates do: helpers.smooth_trace(net_rates, int(rates_dt / dt_sim))
    # smooth out our hyper-resolution rate trace manually cause brian2 can't do it
    results_dict = {'brian_version':2, 'r':smooth_trace(net_rates, ratesdt_dtsim), 't':time_r}
#     results_dict = {'brian_version':2, 'r':net_rates, 't':net_t}
    # print(len(results_dict['t']))
    # time binnig






    if record_v_example_trace > 0:
        results_dict['v'] = v_neurons
        results_dict['t_v'] = t_v


    if record_w > 0 and have_adap:
        results_dict['net_w'] = net_w
        results_dict['net_wt'] = net_wt

    # if record_w > 0 and have_adap:
    #     results_dict['net_w_samples'] = net_w[:min(10, np.size(net_w, 0)), :]
    #     results_dict['wm'] = np.mean(net_w, 0)
    #     # also include
    #     results_dict['w_std'] = np.std(net_w, 0)
    #     results_dict['net_w_dt'] = rates_dt

    if record_spikes > 0:
        results_dict['net_spikes'] = net_spikes
        results_dict['modelname'] = 'net'
    if record_v_stats:
        results_dict['v_mean'] = v_mean
        results_dict['v_var'] = v_var
        results_dict['v_std'] = v_std
    if record_w_stats and have_adap:
        # maybe change this back again ....
        results_dict['wm'] = smooth_trace(w_mean, ratesdt_dtsim)
        results_dict['w_var'] = smooth_trace(w_var, ratesdt_dtsim)
        results_dict['w_std'] = smooth_trace(w_std , ratesdt_dtsim)
    if record_all_v_at_times:
        results_dict['v_all_neurons'] = v_all_neurons
        results_dict['t_all_neurons'] = t_all_neurons
    return results_dict







# version for some parts of the model comparison
# where adaptation is switched on during the run.
#def network_sim_turn_on_adapt(signal, params, rec = False, standalone_dir = cpp_default_dir):
#
#
#    if params['brian2_standalone']:
#        set_device(params['brian2_device'])
#        device.insert_code('main', 'srand('+str(int(time.time())+os.getpid())+');')
#        standalonedir = 'standalone_{}_pid{}'.format(time.strftime("%Y-=%m-%dT%H:%M:%S"), os.getpid())
#    else:
#        prefs.codegen.target = 'numpy'
#
#    #stripe on brian units within the copied dict for the simulation so that brian can work with them
#    dt_sim =        params['net_dt']*ms
#    C =             params['C']*pF
#    gL =            params['gL']*nS
#    taum =          params['taum']*ms
#    EL =            params['EL']*mV
#    Ew =            params['Ew']*mV
#    VT =            params['VT']*mV
#    negVT = -VT
#    deltaT =        params['deltaT']*mV
#    Vcut =          params['Vcut']*mV
#    tauw =          params['tauw']*ms
#    # a =             params['a']*nS
#    # b =             params['b']*pA
#    Vr =            params['Vr']*mV
#    t_ref =         params['t_ref']*ms
#    net_w_init =  params['net_w_init']*pA
#    # this is 1ms
#    rates_dt = params['net_record_dt'] *ms
#    runtime = params['runtime']*ms
#    time_r = np.arange(0., runtime/ms, rates_dt/ms)
#    time_v = np.arange(0., runtime/ms, dt_sim/ms)
#    ratesdt_dtsim=int(rates_dt/dt_sim)
#    N_total = params['N_total']
#    # garbage collect so we can run multiple brian2 runs
#    gc.collect()
#    # seed our random number generator!  necessary on the unix server
#    np.random.seed()
#
#    mu_ext_array = signal[0] # [mV/ms]
#    sigma_ext_array = signal[1] # [mV/sqrt(ms)]
#    if type(mu_ext_array) in [int, float]:
#        mu_ext_array = np.ones_like(time_r)
#    if type(sigma_ext_array) in [int, float]:
#        sigma_ext_array = np.ones_like(time_r)
#
#
#    # extract booleans from params.  unnecessary after above??  at least to please eclipse :)
#    record_spikes = params['net_record_spikes']
#    record_w = params['net_record_w']
#
#    # simulation timestep
#    simclock = Clock(dt_sim)
#
#    w_refr = ' (unless refractory)' if 'net_w_refr' not in params or params['net_w_refr'] else ''
#
#
#
#    print('remove have_adap=True')
#    # instead if/else condition
#    have_adap = True
#    # unpack adaptation params
#    a = params['a']
#    b = params['b']
#    # convert to array if adapt params are scalar values
#    if type(a) in [int,float]:
#        a = np.ones_like(mu_ext_array)*a
#    if type(b) in [int,float]:
#        b = np.ones_like(mu_ext_array)*b
#    a = TimedArray(a*nS,dt_sim)
#    b = TimedArray(b*pA, dt_sim)
#
#
#    #transform the external input into TimedArray
#    mu_ext = TimedArray(mu_ext_array*(mV/ms), dt_sim)
#    sigma_ext = TimedArray(sigma_ext_array*(mV/sqrt(ms)), dt_sim)
#
#    #get the model specific term EIF/PIF
#    # todo: also include LIF model
#    if params['neuron_model'] == 'EIF':
#        model_term = '((EL - v) + deltaT * exp((negVT + v) / deltaT)) / taum'
#    elif params['neuron_model'] == 'PIF':
#        model_term = ''
#    else:
#        mes = 'The model "{}" has not been implemented yet. For options see params dict.'.format(params['neuron_model'])
#        raise NotImplementedError(mes)
#
#    model_eqs = '''
#        dv/dt = %s %s + mu_ext(t) + sigma_ext(t) * xi  : volt (unless refractory)
#        %s
#        ''' % (model_term,'- (w / C)' if have_adap else '', ('dw/dt = (a(t) * (v - Ew) - w) / tauw : amp %s' % w_refr)
#               if have_adap else '')
#
#    # initialize Neuron group
#    G = NeuronGroup(N = N_total, model = model_eqs, threshold = 'v > Vcut', reset = 'v = Vr%s' % ('; w += b(t)' if have_adap else ''),
#                    refractory = t_ref, clock = simclock)
#
#    # initialize rate monitor
#    rate_monitor = PopulationRateMonitor(G, name = 'aeif_ratemon')
#    # intitialize net
#    Net = Network(G, rate_monitor)
#
#    if rec:
#        J = params['J']*mV
#        K = params['K']
#        #synapses object
#        #synapses from G --> G! only one population
#        #this only specifies the dynamics of the synapses.
#        # they get acctually created when the .connect method is called
#        Syn = Synapses(G,G,pre= 'v+=J')
#        sparsity = float(K)/N_total
#        assert 0 <= sparsity <= 1.0
#        # connectivity type
#        if params['connectivity_type'] == 'binomial':
#            Syn.connect(True, p = sparsity)
#        elif params['connectivity_type'] == 'fixed':
#            prelist, postlist = fixed_connectivity(N_total, K)
#            Syn.connect(prelist, postlist)
#        # delays
#        # no delay; nothing has to be implemented
#        if params['delay_type'] == 0:
#            pass
#        # constant delay
#        elif params['delay_type'] == 1:
#            Syn.delay = '{} * ms'.format(params['const_delay'])
#        # exponentially distributed delays
#        elif params['delay_type'] == 2:
#            #taud is the mean=standard deviation of the distribution
#            Syn.delay = '-log(rand()) * {} * ms'.format(params['taud'])
#        # exp. delay dist. + const. delay
#        elif params['delay_type'] == 3:
#            Syn.delay = ('({} -log(rand()) * {}) * ms'.format(format(params['const_delay']), params['taud']))
#        # add synapses to the network
#        Net.add(Syn)
#
#    #initial distribution of the network simulation
#    if params['net_v_init'] == 'delta':
#        G.v = np.ones(len(G)) * params['net_delta_peak']*mV
#    elif params['net_v_init'] == 'normal':
#        G.v = params['net_normal_sigma'] * np.random.randn((len(G))) * mV + params['net_normal_mean'] * mV
#    elif params['net_v_init'] == 'uniform':
#        len_interval = Vcut - Vr
#        G.v = np.random.rand(len(G)) * len_interval + Vr
#    if have_adap:
#        # standart deviation of w_mean is set to 0.1
#        G.w = 0.1 * np.random.randn(len(G)) * pA + net_w_init
#
#    if params['net_record_v_trace']:
#        # define a clock especially for the v_monitor object
#        clock_v_monitor=Clock(dt=505*ms)
#        v_monitor = StateMonitor(G, 'v',clock=clock_v_monitor,  record = True)
#        # add v_monitor to the network in order to record example v-trace
#        Net.add(v_monitor)
#
#    # include a lower bound for the membrane voltage
#    if 'net_v_lower_bound' in params and params['net_v_lower_bound'] is not None:
#        #new in Brian2.0b4: custom_operation --> run_regularly
#        V_lowerbound = G.run_regularly('v = clip(v, %s * mV, 10000 * mV)' % float(params['net_v_lower_bound']),
#                                when = 'end', order = -1, dt = dt_sim)
#        Net.add(V_lowerbound)
#
#    if record_spikes > 0:
#        record_spikes_group = Subgroup(G, 0, min(record_spikes, N_total))
#        spike_monitor = SpikeMonitor(record_spikes_group, name = 'aeif_spikemon')
#        Net.add(spike_monitor, record_spikes_group)
#
#    if record_w > 0 and have_adap:
#        record_w_group = Subgroup(G, 0, min(record_w, N_total))
#        w_monitor = StateMonitor(record_w_group, 'w', record = range(params['net_record_w']), dt = rates_dt, name = 'aeif_wmon')
#        Net.add(w_monitor, record_w_group)
#
#    print('------------------ running network!')
#    start_time = time.time()
#    Net.run(runtime, report = 'text')
#
#    if params['brian2_standalone']:
#        project_dir = cpp_default_dir + '/test' + str(os.getpid())
#        device.build(directory = project_dir, compile=True, run=True)
#
#
#
#    # extract results
#    # rates
#    net_rates = rate_monitor.rate / Hz
#    # example v-trace
#    if params['net_N_v_traces']:
#        v_trace = v_monitor.v/ mV
#        tv_trace = v_monitor.t/ ms
#    if record_w > 0 and have_adap:
#        net_w = w_monitor.w / pA
#    if record_spikes > 0:
#        # multiply by 1 like this to ensure brian extracts the results before we delete the compile directory
#        net_spikes = spike_monitor.it
#        i, t = net_spikes
#        i = i * 1; t = t * 1
#        net_spikes = [i, t]
#
#    run_time = time.time() - start_time
#    print('runtime: %1.1f' % run_time)
#
#
#    if params['brian2_standalone']:
#        shutil.rmtree(project_dir)
#        device.reinit()
#
#    #for smoothing function net_rates do: helpers.smooth_trace(net_rates, int(rates_dt / dt_sim))
#    # smooth out our hyper-resolution rate trace manually cause brian2 can't do it
#
#    results_dict = {'brian_version':2, 'r':smooth_trace(net_rates,ratesdt_dtsim),
#                    'net_rates_dt':rates_dt, 'net_time':run_time, 't':time_r, 't_v':time_v}
#    if params['net_N_v_traces']:
#        results_dict['V'] = v_trace
#        results_dict['tV'] = tv_trace
#    if record_w > 0 and have_adap:
#        results_dict['net_w_samples'] = net_w[:min(10, np.size(net_w, 0)), :]
#        results_dict['wm'] = np.mean(net_w, 0)
#        # also include
#        results_dict['w_std'] = np.std(net_w, 0)
#        results_dict['net_w_dt'] = rates_dt
#    if record_spikes > 0:
#        results_dict['net_spikes'] = net_spikes
#        results_dict['modelname'] = 'net'
#    return results_dict







