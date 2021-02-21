from props import *
import logging


def evaluation_multi_prop(data, metric_type, decode_num=1):
    result = count_multi_prop_score(data)

    if metric_type == 'M6':
        # M6
        count_multi_prop_success_rate(result,
                                      decode_num=1,
                                      sim_delta=0.2,
                                      qed_prop_delta=0.4,
                                      drd2_prop_delta=0.4)
    else:
        raise Exception('metric_type must be chosen from M6! ')


def count_multi_prop_score(data):
    result = []
    for line in data:
        x, y = line.split()
        if y == "None":
            y = None
        sim2D = similarity(x, y)
        try:
            qed_prop = qed(y)
            drd2_prop = drd2(y)
            result.append('%s %s %f %f %f' % (x, y, sim2D, qed_prop, drd2_prop))
        except Exception as e:
            print(e)
            result.append('%s %s %f %f %f' % (x, y, sim2D, 0.0, 0.0))

    return result


def count_multi_prop_success_rate(data, decode_num, sim_delta, qed_prop_delta, drd2_prop_delta):
    logger = logging.getLogger('logger')

    logger.info('==========================================================================================')
    logger.info(
        'Multi Prop SUCCESS RATE: SIM DELTA-%f, QED_DELTA-%f, DRD2_DELTA-%f, NUM DECODE-%d' %
        (sim_delta, qed_prop_delta, drd2_prop_delta, decode_num))

    data = [line.split() for line in data]
    data = [(a, b, float(c), float(d), float(e)) for a, b, c, d, e in data]
    n_mols = len(data) / decode_num
    assert len(data) % decode_num == 0

    n_succ = 0.0
    for i in range(0, len(data), decode_num):
        set_x = set([x[0] for x in data[i:i + decode_num]])
        assert len(set_x) == 1

        good = [(sim, qed_prop, drd2_prop)
                for _, _, sim, qed_prop, drd2_prop in data[i:i + decode_num] if
                1 > sim >= sim_delta and qed_prop >= qed_prop_delta and drd2_prop >= drd2_prop_delta]
        if len(good) > 0:
            n_succ += 1

    logger.info('Evaluated on %d samples' % (n_mols,))
    success_rate = n_succ / n_mols
    logger.info('Multi Prop success rate %f' % success_rate)

    logger.info('==========================================================================================')

    return success_rate
