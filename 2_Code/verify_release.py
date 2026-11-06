"""Verify paper inputs and reported mechanics; run directly from VSCode."""
from pathlib import Path
import ast
import csv
import hashlib
import importlib
import json
import os
import runpy
import sys
import unittest
import numpy as np

# Validation scale; example: True compiles Fortran and checks saved fit costs.
CHECK_COMPILED_UMAT = True
CODE_ROOT = Path(__file__).resolve().parent
os.environ['MPLBACKEND'] = 'Agg'
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (UTILITIES_ROOT as RELEASE_UTILITIES_ROOT,
    INPUT_ROOT as RELEASE_INPUT_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from release_paths import json as release_json
sys.path.insert(0, str(CODE_ROOT.parent))
from package_inventory import package_files
from Analysis.fitting import separable_solver_fit as fitting
from Analysis.core.fit_results import load_fit
from Analysis.studies.model_selection import sv_exp_polycon_mo_component_necessity_study as subsets
from Analysis.diagnostics import sv_exp_polycon_mo_pk1_component_contribution_study as components
from Python_Wrapper import umat
from Python_Wrapper.model_registry import available_model_ids, load_model_spec

def published_input_hashes():
    """Guard measurements and archives; example: plotting must preserve saved SVGs."""
    prefixes = ('1_Experimental_data/', '2_Code/archived_results/')
    return {path.relative_to(CODE_ROOT.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package_files(CODE_ROOT.parent)
            if path.relative_to(CODE_ROOT.parent).as_posix().startswith(prefixes)}

def main():
    """Check syntax, dependency imports, immutable inputs, mechanics, and figures."""
    before = published_input_hashes()
    root = RELEASE_UTILITIES_ROOT
    for file in CODE_ROOT.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
    print('PASS: all Python files parse')
    for file in (root / 'Analysis').rglob('*.py'):
        module = '.'.join(file.relative_to(root).with_suffix('').parts)
        if module.endswith('.__init__'):
            module = module.removesuffix('.__init__')
        importlib.import_module(module)
    print('PASS: every bundled analysis module imports')
    assert available_model_ids() == ('SV_exp_polycon_mo',)
    for file in (RELEASE_INPUT_ROOT / 'Separable_Solver/Cases').glob('*.json'):
        fitting.load_case_configuration(file)
    from Analysis.core import experimental_datasets
    assert len(experimental_datasets.EXPERIMENTAL_DATASETS) == 4
    for key in experimental_datasets.EXPERIMENTAL_DATASETS:
        experimental_datasets.validate_dataset(experimental_datasets.EXPERIMENTAL_DATASETS[key])
    print('PASS: final configurations, legacy metadata reader, and single model registry')
    fits = (1961,1962,1963,1964,2164,2165,2166,2167)
    for fit_folder in sorted((RELEASE_ARCHIVE_ROOT / 'fits').iterdir()):
        code = int(fit_folder.name)
        artifact = fit_folder / "fit.json"
        data = release_json.loads(artifact.read_text())
        fit = load_fit(code)
        assert fit.model == 'SV_exp_polycon_mo'
        for record in data['input_files'].values():
            path = Path(record['snapshot_path'])
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    for config in components.study_configurations():
        components.load_and_validate_fit(config, components.registry_rows())
    inputs = [subsets.load_full_inputs(knit) for knit in subsets.FULL_FIT_CODES]
    print('PASS: eight final fit snapshots and all four nine-ratio datasets')
    suite = unittest.TestSuite()
    for name in ('sv_exp_polycon_mo_component_necessity_study_unit_test.py',
                 'sv_exp_polycon_mo_pk1_component_contribution_study_unit_test.py',
                 'mech_unit_test.py'):
        loaded = runpy.run_path(str(root / 'Unit_Tests' / name))
        for value in loaded.values():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(value))
    runpy.run_path(str(root / 'Unit_Tests/mech_unit_test.py'))['main']()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Bundled mechanics tests failed')
    if CHECK_COMPILED_UMAT:
        umat.compile_umat_meson(load_model_spec('SV_exp_polycon_mo').fortran_source)
        numeric_inputs = []
        for code in fits:
            artifact = release_json.loads((RELEASE_ARCHIVE_ROOT / f'fits/{code}/fit.json').read_text())
            numeric_inputs.append({'knit': artifact['experiment_context']['knit'] + ' fit ' + str(code),
                                   'artifact': artifact,
                                   'parameter_path': Path(artifact['input_files']['material_parameters']['snapshot_path']),
                                   'compiled_path': Path(artifact['input_files']['experimental_data']['snapshot_path'])})
        for item in numeric_inputs:
            wrapper, _ = subsets.configured_full_wrapper(item, compile_source=False)
            for name, value in item['artifact']['parameters'].items():
                wrapper.m.setv_only(name, value)
            # Nonlocal scales must be honored for a short residual subset.
            wrapper.set_cost_function('normalized',1.0,{0:4.0})
            assert np.isclose(wrapper._stress_component_cost(np.array([1.0,2.0]),np.zeros(2),axis=0), (1/5)**2+(2/6)**2)
            references = {axis: float(np.max(np.abs(wrapper.expstressPK2[:,axis]))) for axis in (0,1)}
            wrapper.set_cost_function('normalized',1.0,references)
            cost = wrapper.cost_calculations(axis=[0,1])
            expected = item['artifact']['objective']['raw_cost']
            assert np.isclose(cost,expected,rtol=1e-8,atol=1e-10), (item['knit'],cost,expected)
            print('PASS: compiled UMAT saved cost',item['knit'],cost)
    if CHECK_COMPILED_UMAT:
        from Analysis.studies.solver_diagnostics import separable_solver_coupling_bound_calibration as coupling
        from Analysis.studies.solver_diagnostics import separable_solver_historical_benchmark as historical
        from Python_Wrapper.separable_solver import load_separable_definition
        definition = load_separable_definition('sv_exp_polycon_mo_v1')
        solver = coupling.ShapeCouplingBoundSolver(wrapper,(0,1),definition=definition)
        solver.configure_coupling_bound('amplitude_equalized',0.1)
        shapes = {name: item['artifact']['parameters'][name] for name in definition.shape_parameters}
        solution = solver.solve_linear_at_shapes(shapes)
        parameters = dict(item['artifact']['parameters'])
        parameters.update(solution.linear_values)
        parameters.update(solution.complete_shape_values())
        direct = historical.evaluate_full_parameter_set(wrapper,parameters,(0,1))
        assert np.isclose(direct.raw_cost,solution.cost,rtol=1e-8,atol=1e-10)
        assert solution.cost <= expected + 1e-8
        print('PASS: bounded fixed-shape fit and direct/profiled cost agreement')
    greedy = runpy.run_path(str(CODE_ROOT / '4_Greedy_study/plotter_fig_9_candidates.py'))
    grouped, selected = greedy['load_report']()
    figure, counts = greedy['build_figure'](grouped, selected)
    assert counts == {'main_candidates':171,'negative_candidates':9,'selected':36}
    # Fresh checkouts have no output folders; example: generated/figures.
    figure_directory = RELEASE_GENERATED_ROOT / 'figures'
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_directory / 'verified_figure_9.svg')
    print('PASS: greedy figure reconstructs all 180 candidates and 36 selections')
    # Figure 8 previously overwrote two archived plots; keep this regression visible.
    from Analysis.plotting import plot_single_ratio_calibration_pooled_r2 as single_ratio
    assert single_ratio.OUTPUT_PATH.is_relative_to(RELEASE_GENERATED_ROOT)
    single_ratio.SHOW_PLOT = False
    single_ratio.main()
    ledger = RELEASE_ARCHIVE_ROOT / 'sv_exp_polycon_mo_component_necessity/all_nine_v1/deliverable_hashes.csv'
    with ledger.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            path = ledger.parent / row['relative_path']
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
    assert published_input_hashes() == before, 'Verification changed published measurements or archives'
    print('PASS: Figure 8 generated output, shipped deliverable hashes, and published input preservation')
    print('Release verification passed')

if __name__ == '__main__':
    main()
