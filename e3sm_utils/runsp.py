#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains definitions running ACME-ECP model.
@author: christopher.jones@pnnl.gov
"""

from __future__ import print_function
import os
import subprocess
import xml.etree.ElementTree as ET
import glob
import re
import shlex

# ###########################
# module variables
# ###########################
trial_run = True  # pass test=trial_run default


# ###########################
# Function definitions
# ###########################


def getenv():
    """ Returns HOME directory and HOST name
    """
    home = os.getenv("HOME")
    host = os.getenv("HOST", "None")

    if "edison" in host:
        host = "edison"
    if "blogin" in host:
        host = "anvil"
    if "titan" in host:
        host = "titan"
    if "cori" in host:
        host = "cori-knl"
    return home, host


def max_wallclock_time(nproc):
    """ Maximum wallclock time for nproc/16 nodes on Titan
    """
    if nproc < 125*16:
        return "02:00:00"
    elif nproc < 312*16:
        return "06:00:00"
    elif nproc < 3749*16:
        return "12:00:00"
    else:
        return "24:00:00"


def num_elements(res="ne16"):
    """Calculate number of dynamic and physics elements given res
    """
    if isinstance(res, str):
        # assumes res specified as "neNNN"
        ix = res.find('_')
        ires = int(res[2:ix] if ix > 0 else res[2:])
        return num_elements(ires)
    else:
        num_dynamics = 6 * res * res
        num_physics = 54 * res * res + 2
        return num_dynamics, num_physics


def print_os_command(os_cmd, cwd=None):
    """Print os command to screen (useful for testing)
    """
    # case 1: os_cmd is list in subprocess format [cmd, args]
    if isinstance(os_cmd, list):
        cmd_to_print = " ".join(os_cmd)
        if cwd is not None:
            cmd_to_print = "cd " + cwd + "\n" + cmd_to_print
        cmd_to_execute = os_cmd
    elif isinstance(os_cmd, str):
        # case 2: os_cmd is string "cmd args"
        cmd_to_print = os_cmd
        cmd_to_execute = shlex.split(os_cmd)
    else:
        raise ValueError("os_cmd must be str 'cmd args' or list [cmd, args]")
    if cwd is None:
        subprocess_args = str(cmd_to_execute)
    else:
        subprocess_args = str(cmd_to_execute) + ', cwd=' + cwd
    print("------------------------------------------------------------------")
    print("os cmd: " + cmd_to_print)
    print("python call:")
    print("subprocess.run(" + subprocess_args + ")")
    print("------------------------------------------------------------------")


def common_cam_config_opts(name):
    """Retrieve standard sets of config_opts by name
    """
    opts = {'SP1': {'-use_SPCAM': None,
                    '-rad': 'rrtmg',
                    '-phys': 'cam5',
                    '-nlev': '72',
                    '-crm_nz': '58',
                    '-crm_adv': 'MPDATA',
                    '-crm_nx': '64',
                    '-crm_ny': '1',
                    '-crm_dx': '1000',
                    '-crm_dt': '5',
                    '-SPCAM_microp_scheme': 'sam1mom',
                    '-chem': 'linoz_mam4_resus_mom_soag',
                    '-rain_evap_to_coarse_aero': None,
                    '-bc_dep_to_snow_updates': None,
                    '-cppdefs': "'-DSP_DIR_NS -DAPPLY_POST_DECK_BUGFIXES'"},
            'SP1_nlev30': {'-use_SPCAM': None,
                           '-nlev': '30',
                           '-crm_nz': '28',
                           '-crm_adv': 'MPDATA',
                           '-crm_nx': '64',
                           '-crm_ny': '1',
                           '-crm_dx': '1000',
                           '-crm_dt': '10',
                           '-SPCAM_microp_scheme': 'sam1mom',
                           '-chem': 'linoz_mam4_resus_mom_soag',
                           '-rain_evap_to_coarse_aero': None,
                           '-bc_dep_to_snow_updates': None,
                           '-cppdefs': "'-DSP_DIR_NS'"},
            'SP2_ECPP': {'-use_SPCAM': None,
                         '-rad': 'rrtmg',
                         '-phys': 'cam5',
                         '-nlev': '72',
                         '-crm_nz': '58',
                         '-crm_adv': 'MPDATA',
                         '-crm_nx': '64',
                         '-crm_ny': '1',
                         '-crm_dx': '2000',
                         '-crm_dt': '5',
                         '-microphys': 'mg2',
                         '-SPCAM_microp_scheme': 'm2005',
                         '-chem': 'linoz_mam4_resus_mom_soag',
                         '-rain_evap_to_coarse_aero': None,
                         '-bc_dep_to_snow_updates': None,
                         '-use_ECPP': None,
                         '-cppdefs': "'-DSP_DIR_NS'"},
            'E3SM_RRTMGP': {'-rad': 'rrtmgp',
                            '-phys': 'cam5'}
            }
    if name in opts:
        return opts[name]
    else:
        raise KeyError

# ###########################
# Class definitions
# ###########################


class Config:
    """Store run configuration options
    """
    def __init__(self, newcase=False, config=False, build=False, clean=False,
                 runsim=False, copyinit=False, testrun=False, hindcast=False,
                 continue_run=False, update_namelist=False, debug=False,
                 sp=False):
        self.newcase = newcase                  # configure new case
        self.config = config                    # call case.setup
        self.build = build                      # build the model?
        self.clean = clean                      # clean current
        self.runsim = runsim                    # run the simulation
        self.copyinit = copyinit                # copy files in hindcast run
        self.testrun = testrun                  # test (just print results)
        self.hindcast = hindcast                # run as hindcast
        self.continue_run = continue_run        # continuation run
        self.update_namelist = update_namelist  # update user_nl_cam
        self.debug = debug                      # debug run?
        self.sp = sp                            # use super-parameterization


class Case(object):
    """Class for creating a case
    """
    def __init__(self, case_name,
                 compset="F20TRC5AV1C-04P2",
                 res="ne30",
                 mach="titan",
                 project="m3312",
                 compiler="intel",
                 sp=True,
                 top_dir=None, src_dir=None, scratch_dir=None):
        self.case_name = case_name
        self.compset = compset
        self.res = res
        if mach is None:
            _, mach = getenv()
        self.mach = mach
        self.project = project
        self.compiler = compiler
        self.sp = sp
        self.Directory = AcmeDirectory(case_name=self.case_name,
                                       top_dir=top_dir, src_dir=src_dir,
                                       scratch_dir=scratch_dir)
        self._cdcmd = "cd " + self.Directory.case_dir + " ; "
        self._nproc = None
        self.namelist = {}  # dictionary to hold namelist modifications
        # if self.sp is True:
        #     self.namelist["srf_flux_avg"] = "1"  # always used--should it be?

    def __repr__(self):
        return "Case(%s, compset=%s, res=%s, sp=%s)" % (self.case_name,
                                                        self.compset,
                                                        self.res,
                                                        self.sp)

    def _system_call(self, os_cmd, cwd=None, test=True, verbose=True):
        """Call os_cmd from command line
        """
        if test or verbose:
            print_os_command(os_cmd, cwd=cwd)
        if not test:
            if isinstance(os_cmd, str):
                cmd_to_execute = shlex.split(os_cmd)
            elif isinstance(os_cmd, list):
                cmd_to_execute = os_cmd
            else:
                raise ValueError("os_cmd must be str 'cmd args' or "
                                 "list [cmd, args]")
            subprocess.call(cmd_to_execute, cwd=cwd)

    def delete(self, test=True):
        """ Delete case source and run directories (extreme cleaning)
        """
        acme_dir = self.Directory
        for subdir in [acme_dir.case_dir, acme_dir.scratch_case_dir]:
            if os.path.isdir(subdir):
                self._system_call(['rm', '-r', subdir], test=test)

    def create_newcase(self, test=True, *extra_args):
        """Configure newcase by calling create_newcase cime script
        """
        if "_" in self.res:
            res = self.res
        else:
            res = self.res + "_" + self.res
        cmd = self.Directory.src_dir + "cime/scripts/create_newcase"
        args = ["-case", self.Directory.case_dir,
                "-compset", self.compset,
                "-res", res,
                "-mach", self.Directory.host,
                "-project", self.project,
                "--output-root", self.Directory.scratch_dir]
        if self.compiler is not None:
            args.extend(["-compiler", self.compiler])
        args.extend(extra_args)  # handle any extra args
        self._system_call([cmd] + args, test=test)

    def create_clone(self, OriginalCase, test=True, *extra_args):
        """Clone OriginalCase and point to the original executable
        """
        # be sure to update bld_dir as well:
        self.Directory.bld_dir = OriginalCase.Directory.bld_dir
        cmd = self.Directory.src_dir + "cime/scripts/create_clone"
        args = ["--case", self.Directory.case_dir,
                "--clone", OriginalCase.Directory.case_dir,
                "--keepexe",
                "--project", self.project,
                "--cime-output-root", self.Directory.scratch_dir]
        args.extend(extra_args)
        self._system_call([cmd] + args, test=test)

    def xmlchange(self, variable, value, xmlfile=None, verbose=True, test=True):
        """Make single change variable=value to xmlfile
        """
        if xmlfile is not None:
            os_cmd = "./xmlchange -file " + xmlfile + \
                " -id " + variable + " -val " + value
        else:
            os_cmd = "./xmlchange " + "=".join([variable, value])
        self._system_call(os_cmd, cwd=self.Directory.case_dir,
                          test=test, verbose=verbose)

    def xmlchanges(self, xmlfile=None, verbose=False, test=True, **kwargs):
        """ Make change(s) to xmlfile using keyword args"""
        for key, val in kwargs.items():
            self.xmlchange(key, str(val), xmlfile, verbose=verbose, test=test)

    def xmlread(self, xmlfile):
        """ returns a dictionary of entry id's and values from xmlfile
        """
        fname = self.Directory.case_dir + '/' + xmlfile
        tree = ET.parse(fname)
        root = tree.getroot()
        xml_contents = {}
        for entry in root.iter('entry'):
            name = entry.get('id')
            val = entry.get('value')
            xml_contents[name] = val
        return xml_contents

    def is_build_complete(self):
        xml_contents = self.xmlread('env_build.xml')
        return xml_contents['BUILD_COMPLETE'] == 'TRUE'

    def set_ntasks_in_env_mach_pes(self, num_phys=None, num_dyn=None,
                                   num_threads=None, test=True):
        """Sets NTASKS_ATM=num_phys, NTASKS_XYZ = num_dyn, NTHRDS = num_threads

        If num_dyn or num_phys = None, they are determined by calling
        num_elements(self.res)
        """
        ndyn, nphys = num_elements(self.res)
        if num_phys is None:
            num_phys = nphys
        if num_dyn is None:
            num_dyn = ndyn
        env_mach_pes = {"NTASKS_ATM": str(num_phys),
                        "NTASKS_LND": str(num_dyn),
                        "NTASKS_ICE": str(num_dyn),
                        "NTASKS_OCN": str(num_dyn),
                        "NTASKS_CPL": str(num_dyn),
                        "NTASKS_GLC": str(num_dyn),
                        "NTASKS_ROF": str(num_dyn),
                        "NTASKS_WAV": str(num_dyn),
                        }
        if num_threads is None:
            if self.sp is True:
                env_mach_pes.update(NTHRDS="1")
        else:
            env_mach_pes.update(NTHRDS=num_threads)
        self.xmlchanges("env_mach_pes.xml", test=test, **env_mach_pes)

    def setup(self, clean=False, test=True):
        """Invokes case.setup
        """
        if clean is True:
            self._system_call("./case.setup --clean",
                              cwd=self.Directory.case_dir, test=test)
        else:
            self._system_call("./case.setup",
                              cwd=self.Directory.case_dir, test=test)

    def set_cam_config_opts(self, test=True, default_set=None,
                            dict_pop=None, **kwargs):
        """Sets CAM_CONFIG_OPTS in env_build.xml
        """
        cam_opt_dict = {}
        if default_set is not None:
            cam_opt_dict = common_cam_config_opts(default_set)
        # remove unneeded args from dictionary:
        if dict_pop:
            for key in dict_pop:
                cam_opt_dict.pop(key)
        cam_opt_dict.update(kwargs)  # update dict with new values
        # convert to list of arguments to pass to xmlchange
        cam_opt_args = []
        for key, val in cam_opt_dict.items():
            cam_opt_args.append(key)
            if val:
                cam_opt_args.append(str(val))
        if cam_opt_args:
            xml_contents = {"CAM_CONFIG_OPTS": '"'+' '.join(cam_opt_args)+'"'}
            self.xmlchanges("env_build.xml", test=test, **xml_contents)

    def build(self, clean=False, test=True):
        """Invokes case.build
        """
        if clean is True:
            self._system_call("./case.build --clean",
                              cwd=self.Directory.case_dir, test=test)
        else:
            self._system_call("./case.build",
                              cwd=self.Directory.case_dir, test=test)

    def _default_env_run_config(self):
        env_run = {"RUN_STARTDATE": "2000-01-01",
                   "STOP_OPTION": "ndays",
                   "STOP_N": "5",
                   "RESUBMIT": "0",
                   "CONTINUE_RUN": "FALSE",
                   "INFO_DBUG": "2"}
        return env_run

    def config_env_run(self, test=True, **kwargs):
        """Start from default config; allow user to specify any additional
        kwargs. Updates env_mach_pes.xml (must be called before case.setup)
        """
        xmlfile = "env_run.xml"
        xml_contents = self._default_env_run_config()
        xml_contents.update(kwargs)
        self.xmlchanges(xmlfile, test=test, **xml_contents)

    def config_env_batch(self, test=True, **kwargs):
        """Update env_batch.xml
        """
        xmlfile = "env_batch.xml"
        if self._nproc is None:
            self._nproc = min(num_elements(self.res))  # conservative estimate
        xml_contents = {"JOB_WALLCLOCK_TIME": max_wallclock_time(self._nproc)}
        xml_contents.update(kwargs)
        self.xmlchanges(xmlfile, test=test, **xml_contents)

    def submit(self, test=trial_run, *args):
        """ submit the file
        """
        subfile = "./case.submit"
        self._system_call([subfile] + list(args),
                          cwd=self.Directory.case_dir, test=test)

    def write_namelist_to_file(self, filename=None, namelist=None,
                               write_option='w'):
        """ Writes self.namelist to cam_namelist_file
        """
        if filename is None:
            filename = self.Directory.cam_namelist_file
        if namelist is None:
            namelist = self.namelist
        with open(filename, write_option) as f:
            for key, val in namelist.items():
                line = " " + key + " = " + str(val) + "\n"
                f.write(line)

    def append_namelist_to_file(self, filename=None, **kwargs):
        nl = kwargs if kwargs else None
        self.write_namelist_to_file(filename=filename, namelist=nl,
                                    write_option='a')

    def update_namelist(self, **kwargs):
        self.namelist.update(kwargs)

    def copy_to_casedir(self, file_to_copy, test=True):
        cp_cmd = ["cp"] + [file_to_copy] + [self.Directory.case_dir]
        self._system_call(cp_cmd, test=test)

    def copy_refcase_to_rundir(self, ref_dir, ref_date, ref_case="",
                               test=True, rm_cam_rest=False, pat="*.r*"):
        """ Copy refcase restart files to rundir
        """
        # restart files to copy:
        branch_data = ref_dir + ref_case + pat + ref_date + "*.*"
        files_to_copy = glob.glob(branch_data)
        if files_to_copy:
            cp_cmd = ["cp"] + files_to_copy + [self.Directory.run_dir]
            self._system_call(cp_cmd, test=test)

        # remove cam restart files (optional)
        if rm_cam_rest is True:
            files_to_remove = glob.glob(self.Directory.run_dir + "/" +
                                        ref_case + "*.cam.r*.nc")
            if files_to_remove:
                self._system_call(["rm"] + files_to_remove, test=test)

    def copy_rpointers_to_rundir(self, ref_dir, ref_date,
                                 test=True):
        """ Copy rpointer files to rundir and update with ref_date
        (Necessary for restart runs)
        """
        # rpointer files to update
        branch_data = glob.glob(ref_dir + "rpointer.*")
        for rpointer in branch_data:
            self._system_call(["cp", rpointer, self.Directory.run_dir],
                              test=test)

        # replace date with ref_date:
        pat = r'\.\d{4}-\d{2}-\d{2}'  # date to replace
        filelist = glob.glob(self.Directory.run_dir+"/rpointer*")
        print("Replacing dates in rpointer.* with "+ref_date)
        for FILE in filelist:
            with open(FILE, 'r') as f:
                lines = f.read()
            with open(FILE, 'w') as f:
                f.write(re.sub(pat, '.'+ref_date, lines))


class AcmeDirectory:
    """ Directory structure for ACME-ECP run
    """
    home, host = getenv()

    def __init__(self, case_name, top_dir=None, src_dir=None,
                 scratch_dir=None):
        if top_dir is None:
            top_dir = AcmeDirectory.home + "/git_repos/ACME-ECP/"
        if src_dir is None:
            src_dir = top_dir
        self.top_dir = top_dir
        self.src_dir = src_dir
        self.case_dir = top_dir + "Cases/" + case_name
        self._set_scratch_dir(scratch_dir)
        if self.scratch_dir is not None:
            self.scratch_case_dir = self.scratch_dir+"/"+case_name
            self.bld_dir = self.scratch_case_dir+"/bld"
            self.run_dir = self.scratch_case_dir+"/run"
        self.cam_namelist_file = self.case_dir + "/user_nl_cam"

    def _set_scratch_dir(self, scratch_dir):
        self.scratch_dir = scratch_dir
        if self.scratch_dir is None:
            if "cori" in AcmeDirectory.host:
                self.scratch_dir = os.getenv("CSCRATCH") + "/acme_scratch/cori-knl"
            elif "titan" in AcmeDirectory.host:
                self.scratch_dir = "/lustre/atlas/proj-shared/cli115/" + os.getenv("USER")
            elif "edison" in AcmeDirectory.host:
                self.scratch_dir = os.getenv("CSCRATCH") + "/acme_scratch/edison"


# ###########################
# Variables
# ###########################
compsets = ('F20TRC5AV1C-04P2', 'FC5AV1C-L',
            'FSP1V1', 'FSP2V1', 'FSP1V1-TEST', 'FSP2V1-TEST',
            'F20TRSP1V1', 'F20TRSP2V1')

cam_namelist_for_acme = {
    "ext_frc_cycle_yr": "2000",
    "ext_frc_specifier": """'SO2         -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_elev_2000_c120315.nc',
         'SOAG        -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/aces4bgc_nvsoa_soag_elev_2000_c160427.nc',
         'bc_a4       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_elev_2000_c120315.nc',
         'num_a1      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a1_elev_2000_c120716.nc',
         'num_a2      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_elev_2000_c120315.nc',
         'num_a4      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a3_elev_2000_c120716.nc',
         'pom_a4      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_pom_elev_2000_c130422.nc',
         'so4_a1      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_elev_2000_c120315.nc',
         'so4_a2      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_elev_2000_c120315.nc'""",
    "ext_frc_type": "'CYCLICAL'",
    "srf_emis_cycle_yr": "2000",
    "srf_emis_specifier": """'DMS       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/DMSflux.2000.1deg_latlon_conserv.POPmonthlyClimFromACES4BGC_c20160226.nc',
         'SO2       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_surf_2000_c120315.nc',
         'bc_a4     -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_surf_2000_c120315.nc',
         'num_a1    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a1_surf_2000_c120716.nc',
         'num_a2    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_surf_2000_c120315.nc',
         'num_a4    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a3_surf_2000_c120716.nc',
         'pom_a4    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_pom_surf_2000_c130422.nc',
         'so4_a1    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_surf_2000_c120315.nc',
         'so4_a2    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_surf_2000_c120315.nc'""",
    "srf_emis_type": "'CYCLICAL'",
    "tracer_cnst_cycle_yr": "2000",
    "tracer_cnst_datapath": "'/lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/oxid'",
    "tracer_cnst_file": "'oxid_1.9x2.5_L26_1850-2005_c091123.nc'",
    "tracer_cnst_filelist": "'oxid_1.9x2.5_L26_clim_list.c090805.txt'",
    "tracer_cnst_specifier": "'cnst_O3:O3','OH','NO3','HO2'",
    "tracer_cnst_type": "'CYCLICAL'",
    "prescribed_ozone_cycle_yr": "2000",
    "prescribed_ozone_type": "'CYCLICAL'"
}

cam_namelist_for_nudging = {
    "Nudge_Model": ".True.",
    "Nudge_Path": "'/lustre/atlas/proj-shared/csc249/crjones/era-interim/ne30/'",
    "Nudge_File_Template": "'regrid_IE_ne30.cam2.i.%y-%m-%d-%s.nc'",
    "Nudge_Times_Per_Day": "4",
    "Model_Times_Per_Day": "48",
    "Nudge_Uprof": "1",
    "Nudge_Ucoef": "1.",
    "Nudge_Vprof": "1",
    "Nudge_Vcoef": "1.",
    "Nudge_Tprof": "0",
    "Nudge_Tcoef": "0.",
    "Nudge_Qprof": "0",
    "Nudge_Qcoef": "0.",
    "Nudge_PSprof": "0",
    "Nudge_PScoef": "0.",
    "Nudge_Beg_Year": "0000",
    "Nudge_Beg_Month": "1",
    "Nudge_Beg_Day": "1",
    "Nudge_End_Year": "9999",
    "Nudge_End_Month": "1",
    "Nudge_End_Day": "1",
    "inithist": "'DAILY'",
    "ext_frc_cycle_yr": "2000",
    "ext_frc_specifier": """'SO2         -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_elev_2000_c120315.nc',
         'SOAG        -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/aces4bgc_nvsoa_soag_elev_2000_c160427.nc',
         'bc_a4       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_elev_2000_c120315.nc',
         'num_a1      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a1_elev_2000_c120716.nc',
         'num_a2      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_elev_2000_c120315.nc',
         'num_a4      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a3_elev_2000_c120716.nc',
         'pom_a4      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_pom_elev_2000_c130422.nc',
         'so4_a1      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_elev_2000_c120315.nc',
         'so4_a2      -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_elev_2000_c120315.nc'""",
    "ext_frc_type": "'CYCLICAL'",
    "srf_emis_cycle_yr": "2000",
    "srf_emis_specifier": """'DMS       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/DMSflux.2000.1deg_latlon_conserv.POPmonthlyClimFromACES4BGC_c20160226.nc',
         'SO2       -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so2_surf_2000_c120315.nc',
         'bc_a4     -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_bc_surf_2000_c120315.nc',
         'num_a1    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a1_surf_2000_c120716.nc',
         'num_a2    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_num_a2_surf_2000_c120315.nc',
         'num_a4    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam7_num_a3_surf_2000_c120716.nc',
         'pom_a4    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_pom_surf_2000_c130422.nc',
         'so4_a1    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a1_surf_2000_c120315.nc',
         'so4_a2    -> /lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/emis/ar5_mam3_so4_a2_surf_2000_c120315.nc'""",
    "srf_emis_type": "'CYCLICAL'",
    "tracer_cnst_cycle_yr": "2000",
    "tracer_cnst_datapath": "'/lustre/atlas/world-shared/cli900/cesm/inputdata/atm/cam/chem/trop_mozart_aero/oxid'",
    "tracer_cnst_file": "'oxid_1.9x2.5_L26_1850-2005_c091123.nc'",
    "tracer_cnst_filelist": "'oxid_1.9x2.5_L26_clim_list.c090805.txt'",
    "tracer_cnst_specifier": "'cnst_O3:O3','OH','NO3','HO2'",
    "tracer_cnst_type": "'CYCLICAL'",
    "prescribed_ozone_cycle_yr": "2000",
    "prescribed_ozone_type": "'CYCLICAL'"
}

cam_namelist_hindcast_sp = {
    "iradsw": "1",
    "iradlw": "1",
    "nhtfrq": "0,-1,-1,-1,-1",
    "mfilt": "1,24,24,24,1",
    "fincl2": "'T','Q','Z3','OMEGA','U','V','CLOUD'",
    "fincl3": "'TS','TMQ','PRECT','TREFHT','LHFLX','SHFLX',"
              "'FLNS','FLNT','FSNS','FSNT','FLUT',"
              "'CLDLOW','CLDMED','CLDHGH','CLDTOT',"
              "'U850','U200','V850','V200','OMEGA500',"
              "'LWCF','SWCF','PS','PSL','QAP:A','TAP:A','PRECC','PRECL'",
    "fincl4": "'DTCORE:A','SPDT:A','SPDQ:A','PTTEND:A','PTEQ:A',"
              "'DTV:A','VD01:A','QRL:A','QRS:A','QAP:I','TAP:I',"
              "'SPQC','SPQI','SPQS','SPQR','SPQG','SPQPEVP'"}

cam_namelist_hindcast_acme = {
    "nhtfrq": "0,-1,-1,-1",
    "mfilt": "1,24,24,24",
    "fincl2": "'T','Q','Z3','OMEGA','U','V','CLOUD'",
    "fincl3": "'TS','TMQ','PRECT','TREFHT','LHFLX','SHFLX',"
              "'FLNS','FLNT','FSNS','FSNT','FLUT',"
              "'CLDLOW','CLDMED','CLDHGH','CLDTOT',"
              "'U850','U200','V850','V200','OMEGA500',"
              "'LWCF','SWCF','PS','PSL','QAP:A','TAP:A','PRECC','PRECL'",
    "fincl4": "'DTCORE:A','PTTEND:A','PTEQ:A',"
              "'QRL:A','QRS:A','QAP:I','TAP:I'",
    "empty_htapes": ".true."}


def build_namelist(sp=True, hindcast=True, nudge=False):
    """Build typical namelists
    """
    cam_namelist = {}
    if nudge:
        cam_namelist.update(**cam_namelist_for_nudging)
    if sp is False:
        cam_namelist.update(**cam_namelist_for_acme)
    if hindcast:
        if sp:
            cam_namelist.update(**cam_namelist_hindcast_sp)
        else:
            cam_namelist.update(**cam_namelist_hindcast_acme)
    return cam_namelist
