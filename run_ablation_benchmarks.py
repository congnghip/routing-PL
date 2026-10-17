import os
import sys
import subprocess
import csv

instances_order = [
    'C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109',
    'C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208',
    'R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112',
    'R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211',
    'RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108',
    'RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208'
]

ROUTING_DIR = '/home/congnghip/project/routing-PL'
PAPER_DIR = '/home/congnghip/project/academic-papers/papers/2026_vrppl_ig_vs_gpu'
IG_FILE = '/home/congnghip/project/prj_vrppl-main/Docs/latex_parser/result_100.csv'

def evaluate_and_generate_table(csv_path, tex_out_path, caption_title, tab_label):
    with open(csv_path) as f:
        gpu = {r['instance_name'].replace('_co_100.txt', ''): r for r in csv.DictReader(f)}
    with open(IG_FILE) as f:
        pub = {r['instance']: r for r in csv.DictReader(f)}

    lines = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption_title}}}')
    lines.append(f'\\label{{{tab_label}}}')
    lines.append('\\footnotesize')
    lines.append('\\renewcommand{\\arraystretch}{1.12}')
    lines.append('\\setlength{\\tabcolsep}{4.2pt}')
    lines.append('\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrrr@{}}')
    lines.append('\\toprule')
    lines.append('\\multirow{2}{*}{\\textbf{Instance}} & \\multicolumn{4}{c}{\\textbf{IG (Corr\\^ea et al., 2026)}} & \\multicolumn{4}{c}{\\textbf{GPU 3D-PACO (Ours)}} & \\multirow{2}{*}{\\textbf{GAP (\\%)}} \\\\')
    lines.append('\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}')
    lines.append(' & \\textbf{Avg. Cost} & \\textbf{Avg. Time (s)} & \\textbf{Best Cost} & \\textbf{NV} & \\textbf{Avg. Cost} & \\textbf{Avg. Time (s)} & \\textbf{Best Cost} & \\textbf{NV} & \\\\')
    lines.append('\\midrule')

    sum_ig_avg = 0.0
    sum_ig_time = 0.0
    sum_ig_best = 0.0
    sum_ig_nv = 0.0
    
    sum_gpu_avg = 0.0
    sum_gpu_time = 0.0
    sum_gpu_best = 0.0
    sum_gpu_nv = 0.0
    
    better_count = 0
    equal_count = 0
    worse_count = 0

    for inst in instances_order:
        ig_r = pub[inst]
        gpu_r = gpu[inst]
        
        ig_avg_c = float(ig_r['ig_avg_cost'])
        ig_avg_t = float(ig_r['ig_avg_time'])
        ig_best_c = float(ig_r['ig_best_cost'])
        ig_nv = int(ig_r['ig_n_v'])
        
        gpu_avg_c = float(gpu_r['AVG Distance'])
        gpu_avg_t = float(gpu_r['AVG Runtime (s)'])
        gpu_best_c = float(gpu_r['Best Distance'])
        gpu_nv = int(gpu_r['Num Vehicles'])
        
        sum_ig_avg += ig_avg_c
        sum_ig_time += ig_avg_t
        sum_ig_best += ig_best_c
        sum_ig_nv += ig_nv
        
        sum_gpu_avg += gpu_avg_c
        sum_gpu_time += gpu_avg_t
        sum_gpu_best += gpu_best_c
        sum_gpu_nv += gpu_nv
        
        gap = (gpu_best_c - ig_best_c) / ig_best_c * 100.0
        
        diff = round(gpu_best_c - ig_best_c, 3)
        if diff < -0.01:
            better_count += 1
        elif abs(diff) <= 0.01:
            equal_count += 1
        else:
            worse_count += 1
            
        ig_avg_c_s = f'\\textbf{{{ig_avg_c:.3f}}}' if ig_avg_c <= gpu_avg_c else f'{ig_avg_c:.3f}'
        gpu_avg_c_s = f'\\textbf{{{gpu_avg_c:.3f}}}' if gpu_avg_c < ig_avg_c else f'{gpu_avg_c:.3f}'
            
        ig_avg_t_s = f'\\textbf{{{ig_avg_t:.3f}}}' if ig_avg_t <= gpu_avg_t else f'{ig_avg_t:.3f}'
        gpu_avg_t_s = f'\\textbf{{{gpu_avg_t:.3f}}}' if gpu_avg_t < ig_avg_t else f'{gpu_avg_t:.3f}'
            
        if gpu_best_c < ig_best_c:
            ig_best_c_s = f'{ig_best_c:.3f}'
            gpu_best_c_s = f'\\textbf{{{gpu_best_c:.3f}}}'
        elif ig_best_c < gpu_best_c:
            ig_best_c_s = f'\\textbf{{{ig_best_c:.3f}}}'
            gpu_best_c_s = f'{gpu_best_c:.3f}'
        else:
            ig_best_c_s = f'\\textbf{{{ig_best_c:.3f}}}'
            gpu_best_c_s = f'\\textbf{{{gpu_best_c:.3f}}}'
            
        if gpu_nv < ig_nv:
            ig_nv_s = str(ig_nv)
            gpu_nv_s = f'\\textbf{{{gpu_nv}}}'
        elif ig_nv < gpu_nv:
            ig_nv_s = f'\\textbf{{{ig_nv}}}'
            gpu_nv_s = str(gpu_nv)
        else:
            ig_nv_s = str(ig_nv)
            gpu_nv_s = str(gpu_nv)
            
        gap_s = f'\\textbf{{{gap:.3f}}}' if gap < 0 else f'{gap:.3f}'
        
        lines.append(f'{inst} & {ig_avg_c_s} & {ig_avg_t_s} & {ig_best_c_s} & {ig_nv_s} & {gpu_avg_c_s} & {gpu_avg_t_s} & {gpu_best_c_s} & {gpu_nv_s} & {gap_s} \\\\')

    n_inst = len(instances_order)
    avg_ig_avg = sum_ig_avg / n_inst
    avg_ig_time = sum_ig_time / n_inst
    avg_ig_best = sum_ig_best / n_inst
    avg_ig_nv = sum_ig_nv / n_inst

    avg_gpu_avg = sum_gpu_avg / n_inst
    avg_gpu_time = sum_gpu_time / n_inst
    avg_gpu_best = sum_gpu_best / n_inst
    avg_gpu_nv = sum_gpu_nv / n_inst

    avg_gap = (avg_gpu_best - avg_ig_best) / avg_ig_best * 100.0

    lines.append('\\midrule')
    lines.append(f'\\textbf{{Avg.}} & {avg_ig_avg:.3f} & \\textbf{{{avg_ig_time:.3f}}} & {avg_ig_best:.3f} & {avg_ig_nv:.2f} & \\textbf{{{avg_gpu_avg:.3f}}} & {avg_gpu_time:.3f} & \\textbf{{{avg_gpu_best:.3f}}} & \\textbf{{{avg_gpu_nv:.2f}}} & \\textbf{{{avg_gap:.3f}}} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular*}')
    lines.append('\\end{table*}')

    with open(tex_out_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return {
        'ig_best': avg_ig_best,
        'gpu_best': avg_gpu_best,
        'ig_avg': avg_ig_avg,
        'gpu_avg': avg_gpu_avg,
        'ig_nv': avg_ig_nv,
        'gpu_nv': avg_gpu_nv,
        'ig_time': avg_ig_time,
        'gpu_time': avg_gpu_time,
        'gap': avg_gap,
        'better': better_count,
        'equal': equal_count,
        'worse': worse_count
    }

def main():
    experiments = [
        ('Baseline (No Levers)', [], 'large_gpu_baseline.csv', 'table_large_baseline.tex', 'tab:vrppl_large_baseline'),
        ('Lever 1 (Intensified LS)', ['--lever1'], 'large_gpu_lever1.csv', 'table_large_lever1.tex', 'tab:vrppl_large_lever1'),
        ('Lever 2 (Pheromone R&R)', ['--lever2'], 'large_gpu_lever2.csv', 'table_large_lever2.tex', 'tab:vrppl_large_lever2'),
        ('Lever 3 (Inter-Route Operators)', ['--lever3'], 'large_gpu_lever3.csv', 'table_large_lever3.tex', 'tab:vrppl_large_lever3'),
        ('Pair 1+2 (Intensified LS + Pheromone R&R)', ['--lever1', '--lever2'], 'large_gpu_lever12.csv', 'table_large_lever12.tex', 'tab:vrppl_large_lever12'),
        ('Pair 1+3 (Intensified LS + Inter-Route)', ['--lever1', '--lever3'], 'large_gpu_lever13.csv', 'table_large_lever13.tex', 'tab:vrppl_large_lever13'),
        ('Pair 2+3 (Pheromone R&R + Inter-Route)', ['--lever2', '--lever3'], 'large_gpu_lever23.csv', 'table_large_lever23.tex', 'tab:vrppl_large_lever23'),
        ('Full 1+2+3 (All Three Levers Combined)', ['--lever1', '--lever2', '--lever3'], 'large_gpu_lever123.csv', 'table_large_lever123.tex', 'tab:vrppl_large_lever123')
    ]

    all_results = []

    for name, flags, csv_name, tex_name, tab_label in experiments:
        csv_path = os.path.join(ROUTING_DIR, 'src/output/solutions', csv_name)
        tex_path = os.path.join(PAPER_DIR, tex_name)
        
        print(f"\n====================================================================")
        print(f"  RUNNING CONFIGURATION: {name}")
        print(f"====================================================================")
        
        # Build command
        cmd = [os.path.join(ROUTING_DIR, 'src/cuda/paco_gpu_solver'), '100'] + flags + ['--out', csv_path]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        caption = f"Computational comparison between IG (Corr\\^ea et al., 2026) and GPU 3D-PACO under \\textbf{{{name}}} on Large ($n=100$) instances."
        stats = evaluate_and_generate_table(csv_path, tex_path, caption, tab_label)
        stats['name'] = name
        stats['csv'] = csv_name
        stats['tex'] = tex_name
        all_results.append(stats)
        
        print(f"--- [Benchmark Result for {name}] ---")
        print(f"  Best Cost : GPU = {stats['gpu_best']:.3f} vs IG = {stats['ig_best']:.3f} | GAP = {stats['gap']:+.3f}%")
        print(f"  Mean Cost : GPU = {stats['gpu_avg']:.3f} vs IG = {stats['ig_avg']:.3f}")
        print(f"  Fleet NV  : GPU = {stats['gpu_nv']:.2f} vs IG = {stats['ig_nv']:.2f}")
        print(f"  Mean Time : GPU = {stats['gpu_time']:.4f} s vs IG = {stats['ig_time']:.4f} s")
        print(f"  Wins/Ties/Losses : {stats['better']} Won / {stats['equal']} Tied / {stats['worse']} Lost")
        print(f"  Exported LaTeX table to: {tex_path}")

    # Generate Grand Summary Table
    sum_lines = []
    sum_lines.append('\\begin{table*}[t]')
    sum_lines.append('\\centering')
    sum_lines.append('\\caption{Ablation Study of Algorithmic Levers on Large ($n=100$) Instances: Progressive Comparison against Iterated Greedy (IG, Corr\\^ea et al., 2026).}')
    sum_lines.append('\\label{tab:vrppl_ablation_summary}')
    sum_lines.append('\\footnotesize')
    sum_lines.append('\\renewcommand{\\arraystretch}{1.2}')
    sum_lines.append('\\setlength{\\tabcolsep}{5pt}')
    sum_lines.append('\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}')
    sum_lines.append('\\toprule')
    sum_lines.append('\\textbf{Algorithmic Configuration} & \\textbf{Best Cost} & \\textbf{Mean Cost} & \\textbf{NV} & \\textbf{Time (s)} & \\textbf{Cost GAP (\\%)} & \\textbf{GPU $<$ IG} & \\textbf{GPU $\\ge$ IG} \\\\')
    sum_lines.append('\\midrule')
    
    # Baseline IG row
    ig_b = all_results[0]['ig_best']
    ig_a = all_results[0]['ig_avg']
    ig_v = all_results[0]['ig_nv']
    ig_t = all_results[0]['ig_time']
    sum_lines.append(f"\\textbf{{IG Baseline (Corr\\^ea et al., 2026)}} & {ig_b:.3f} & {ig_a:.3f} & {ig_v:.2f} & {ig_t:.4f} & 0.000\\% & -- & -- \\\\")
    sum_lines.append('\\midrule')

    for r in all_results:
        gap_s = f"\\textbf{{{r['gap']:+.3f}\\%}}" if r['gap'] < 0 else f"{r['gap']:+.3f}\\%"
        sum_lines.append(f"{r['name']} & {r['gpu_best']:.3f} & {r['gpu_avg']:.3f} & {r['gpu_nv']:.2f} & {r['gpu_time']:.4f} & {gap_s} & \\textbf{{{r['better']}}} & {r['equal'] + r['worse']} \\\\")
    
    sum_lines.append('\\bottomrule')
    sum_lines.append('\\end{tabular*}')
    sum_lines.append('\\end{table*}')

    summary_tex = os.path.join(PAPER_DIR, 'table_ablation_summary.tex')
    with open(summary_tex, 'w') as f:
        f.write('\n'.join(sum_lines))
    print(f"\nSuccessfully generated Grand Ablation Table at: {summary_tex}")

if __name__ == '__main__':
    main()
