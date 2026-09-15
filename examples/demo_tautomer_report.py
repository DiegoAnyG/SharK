#!/usr/bin/env python3
"""Generate thermochemistry, spectra and 3D frontier reports for an ORCA directory."""

import argparse
import json
from pathlib import Path

from shark.core.parser import parse_orca_results
from shark.core.frontier import discover_frontier_fields
from shark.analysis.thermo import calculate_relative_thermo
from shark.reports.visualizer import (
    plot_boltzmann_equilibrium, plot_ir_comparison,
    generate_markdown_report, generate_html_report,
)
from shark.reports.orbital_viewer import ViewSettings, export_orbital_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='ORCA calculation directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Orbital settings and optional calculation manifest')
    parser.add_argument('--temperature', type=float, default=298.15)
    parser.add_argument('--html-only', action='store_true', help='Skip static orbital export if Chrome is unavailable')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    specs = config.get('calculations')
    if specs is None:
        specs = [dict(output=p.name, label=p.stem.replace('_', ' ')) for p in sorted(args.input.glob('*.out'))]
    if not specs:
        parser.error('No ORCA output files found')
    if not args.output.exists():
        print(f'Creating {args.output}: generated reports, safe to remove and regenerate.')
    args.output.mkdir(parents=True, exist_ok=True)
    results = [parse_orca_results(args.input/s['output'], name=s.get('label')) for s in specs]
    data = calculate_relative_thermo(results, temperature=args.temperature)
    data.to_csv(args.output/'thermodynamics_summary.csv', index=False)
    images = [plot_boltzmann_equilibrium(data, args.output/'boltzmann_distribution.png'),
              plot_ir_comparison(results, args.output/'ir_spectra_comparison.png')]
    orbital_files = export_orbital_report(
        discover_frontier_fields(args.input, specs), args.output,
        ViewSettings(**config.get('settings', {})), formats=('html',) if args.html_only else ('html', 'png', 'pdf'))
    if 'png' in orbital_files:
        images.append(orbital_files['png'])
    generate_markdown_report(data, args.output/'report.md', notes=(
        'Orbital figures and their reproducibility metadata are provided alongside this report. '
        'Inspect convergence and stationary-point status for each calculation before interpreting populations.'))
    generate_html_report(data, images, args.output/'report.html', interactive_reports=[orbital_files['html']])
    print(f'Report: {args.output / "report.html"}')


if __name__ == '__main__':
    main()
