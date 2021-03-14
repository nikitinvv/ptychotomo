# set @profile decorator for functions you want to profile
#kernprof -l test_tomo.py
#python -m line_profiler test_tomo.py.lprof > test_tomo.py.lineprof

#kernprof -l test_deform.py
#python -m line_profiler test_deform.py.lprof > test_deform.py.lineprof


# kernprof -l test_perf.py
# python -m line_profiler test_perf.py.lprof > test_perf.py.lineprof

kernprof -l test_ptycho_cg.py
python -m line_profiler test_ptycho_cg.py.lprof > test_ptycho_cg.py.lineprof
