import numpy as np
from minepy import MINE
from multiprocessing import Pool


def analyse_mic(n, m, x, y):
    mine = MINE(alpha=0.6, c=16, est="mic_approx")
    # print("MINE(alpha=0.6, c=16, est=mic_approx)", flush=True)

    # mine = MINE(alpha=0.6, c=16, est="mic_e")
    # print("MINE(alpha=0.6, c=16, est=mic_e)", flush=True)
    mine.compute_score(x, y)
    return n, m, mine.mic(), mine.mas(), mine.mev(), mine.mcn(0), mine.mcn_general(), mine.gmic(), mine.tic()


def scorr():
    pool = Pool(processes=14)

    data_path_list = ["../data/PEMS03/PEMS03.npz", "../data/PEMS04/PEMS04.npz", "../data/PEMS07/PEMS07.npz", "../data/PEMS08/PEMS08.npz",
                      "../data/HZME_INFLOW/HZME_INFLOW.npz", "../data/HZME_OUTFLOW/HZME_OUTFLOW.npz"]
    data_name_list = ["PEMS03", "PEMS04", "PEMS07", "PEMS08", "HZME_INFLOW", "HZME_OUTFLOW"]

    for i in range(len(data_name_list)):
        data_path = data_path_list[i]
        data_name = data_name_list[i]

        data = np.load(data_path)['data']  # [T,N,F]
        T, N, F = data.shape
        data = data[:int(T * 0.6), :, :]
        result = []

        mic_matrix = np.zeros((F, N, N))
        mas_matrix = np.zeros((F, N, N))
        mev_matrix = np.zeros((F, N, N))
        mcn_matrix = np.zeros((F, N, N))
        mcn_general_matrix = np.zeros((F, N, N))
        gmic_matrix = np.zeros((F, N, N))
        tic_matrix = np.zeros((F, N, N))

        for c in range(F):
            for n_index in range(N):
                for m_index in range(n_index, N):
                    print(data_name, n_index, m_index, flush=True)
                    res = pool.apply_async(analyse_mic, args=(n_index, m_index, data[:, n_index, c], data[:, m_index, c]))
                    result.append(res)

            for r in result:
                n_index, m_index, mic, mas, mev, mcn, mcn_general, gmic, tic = r.get()
                mic_matrix[c, n_index, m_index] = mic
                mic_matrix[c, m_index, n_index] = mic

                mas_matrix[c, n_index, m_index] = mas
                mas_matrix[c, m_index, n_index] = mas

                mev_matrix[c, n_index, m_index] = mev
                mev_matrix[c, m_index, n_index] = mev

                mcn_matrix[c, n_index, m_index] = mcn
                mcn_matrix[c, m_index, n_index] = mcn

                mcn_general_matrix[c, n_index, m_index] = mcn_general
                mcn_general_matrix[c, m_index, n_index] = mcn_general

                gmic_matrix[c, n_index, m_index] = gmic
                gmic_matrix[c, m_index, n_index] = gmic

                tic_matrix[c, n_index, m_index] = tic
                tic_matrix[c, m_index, n_index] = tic

                print(n_index, m_index)

        np.savez("All_" + data_name,
                 mic_matrix=mic_matrix,
                 mas_matrix=mas_matrix,
                 mev_matrix=mev_matrix,
                 mcn_matrix=mcn_matrix,
                 mcn_general_matrix=mcn_general_matrix,
                 gmic_matrix=gmic_matrix,
                 tic_matrix=tic_matrix)
        np.save("SCORR_" + data_name, mic_matrix)


def tcorr_x(data, node_index, num_of_xx, base, weight):
    """

    Parameters
    ----------
    data
    node_index: the index of node
    num_of_xx: the number of hour, day or week ago, such as 1 hour ago, 1 day ago, 1 week ago
    base: the time points in hour, day or week, including 12, 12*24, 12*24*7

    Returns
    -------

    """
    data_start = num_of_xx * base
    data_end = data.shape[0] - 12
    length = data_end - data_start + 1
    mic_sum, mas_sum, mev_sum, mcn_sum, mcn_general_sum, gmic_sum, tic_sum = 0, 0, 0, 0, 0, 0, 0

    del_count = 0
    for index in range(length):
        target_data_start = index + data_start
        target = data[target_data_start:target_data_start + 12][:, 0]

        sample_data_start = index + data_start - num_of_xx * base
        sample = data[sample_data_start:sample_data_start + 12, 0]

        if target_data_start % (24 * 12) > 6 * 12:  # the metro crowd flow datasets only contain zero between 0:00 and 6:00, delete the part
            l_index, n_index, mic, mas, mev, mcn, mcn_general, gmic, tic = analyse_mic(index, node_index, sample, target)
            mic_sum += mic * weight
            mas_sum += mas * weight
            mev_sum += mev * weight
            mcn_sum += mcn * weight
            mcn_general_sum += mcn_general * weight
            gmic_sum += gmic * weight
            tic_sum += tic * weight
        else:
            del_count += 1

    length -= del_count
    # print(length)
    return mic_sum / length, mas_sum / length, mev_sum / length, mcn_sum / length, mcn_general_sum / length, gmic_sum / length, tic_sum / length


def tcorr_hdw(data, node_index):
    test_num = 2
    result = np.zeros((3, test_num, 7))
    # for week data
    for num_of_week in range(test_num):
        result[0, num_of_week, 0], result[0, num_of_week, 1], result[0, num_of_week, 2], result[0, num_of_week, 3], result[0, num_of_week, 4], result[0, num_of_week, 5], result[0, num_of_week, 6] = \
            tcorr_x(data, node_index, num_of_week, 7 * 24 * 12, weight=0.85)
    for num_of_day in range(test_num):
        result[1, num_of_day, 0], result[1, num_of_day, 1], result[1, num_of_day, 2], result[1, num_of_day, 3], result[1, num_of_day, 4], result[1, num_of_day, 5], result[1, num_of_day, 6] = \
            tcorr_x(data, node_index, num_of_day, 24 * 12, weight=0.95)
    for num_of_hour in range(test_num):
        result[2, num_of_hour, 0], result[2, num_of_hour, 1], result[2, num_of_hour, 2], result[2, num_of_hour, 3], result[2, num_of_hour, 4], result[2, num_of_hour, 5], result[2, num_of_hour, 6] = \
            tcorr_x(data, node_index, num_of_hour, 12, weight=0.95)
    return node_index, result


def tcorr():
    pool = Pool(processes=14)

    data_path_list = ["../data/PEMS03/PEMS03.npz", "../data/PEMS04/PEMS04.npz", "../data/PEMS07/PEMS07.npz", "../data/PEMS08/PEMS08.npz",
                      "../data/HZME_INFLOW/HZME_INFLOW.npz", "../data/HZME_OUTFLOW/HZME_OUTFLOW.npz"]
    data_name_list = ["PEMS03", "PEMS04", "PEMS07", "PEMS08", "HZME_INFLOW", "HZME_OUTFLOW"]

    for i in range(len(data_name_list)):
        data_path = data_path_list[i]
        data_name = data_name_list[i]
        data = np.load(data_path)['data'][:int(T * 0.6), :, :1]  # [T,N,F] train data
        T, N, D = data.shape
        data = np.load(data_path)['data']  # [T,N,F]
        T, N, F = data.shape
        data = data[:int(T * 0.6), :, :1]

        test_num = 2
        result = np.zeros((N, 3, test_num, 7))
        result_multiprocess = []
        for node_index in range(N):
            res = pool.apply_async(tcorr_hdw, args=(data[:, node_index, :], node_index))
            result_multiprocess.append(res)
            # analyse_time_by_time(data[:, node_index, :], node_index)
        for re_m in result_multiprocess:
            node_index, result_node = re_m.get()
            print(data_name, node_index, flush=True)
            result[node_index, :, :, :] = result_node

        np.save("TCORR_" + data_name, result)


if __name__ == '__main__':
    scorr()
    tcorr()
