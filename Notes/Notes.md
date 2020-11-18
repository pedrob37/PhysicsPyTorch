# Opening atom/ other useful Linux things
- /etc/init.d/vmware-tools restart ~ To gain access to shared folders
- Start new terminal (e.g. Desktop) and open from there, NOT root path

# Shared folders
- https://communities.vmware.com/thread/450790 ~ Walkthrough on activation

# Troubleshooting
- If FSLeyes breaks: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes (Standalone installation)
- If access to SFolder is denied (i.e. shortcut no longer exists error message) then: 1. Re-install vmtools
                                                                                     (2. Could maybe just re-run vmware-tools-config.pl)
                                                                                     (3. Because of kernel changes?)

- If chrome shortcut stops working: 1. Try re-installing: sudo yum install google-chrome-stable
                                    2. Can run using google-chrome &
                                    3. Navigate to <usr/share/applications> + xdg-open .
                                    4. Right-click chrome (or whatever) and copy-to Desktop
                                    5. Execute and Trust icon in Desktop

# Git docs.
- git commit --amend: -> Edit -> [Esc] + [:wq] + [Enter]

# Searching on Linux/ Other useful linux commands
- sudo find / -name 'NAME'
  - Deeper: find . -name [FILE] -type f
- <SEARCH> {CTRL+R} [PATTERN] ~ Repeat [PATTERN] to keep searching
- <SEARCH> history | [PATTERN] ~ Displays list of where [PATTERN] is found

- <ROOT> [su -] ~ Enter root
- <PATH> export PATH=$PATH:/bin:[DIRECTORY] ~ Used this to add possum_working to useable commands
  - For permanent addition, change ~/.bash_profile, and add [PATH=$PATH:/bin:DIRECTORY] to it
- <EXECUTABLE> chmod +x [FILENAME] ~ Make sh file executable
- <RELOAD> [source FILE] ~ Evaluates file, e.g. source ~/.bash_profile
- <RESOLUTION> [xrandr -q] ~ Check resolutions. See {/home/fsluser/resolution.sh} file for more details
  - Execute using [source ./resolution.sh]
- <DIRSIZE> [du -sh DIR] ~ Evaluates size of specified directory  ##SIZE
            [du -sh */]  ~ List all directories in current directory and show size of each
- <DIRSIZES> [du --max-depth=1 | sort -rn] ~ Orders directories in current directory in descending order of size (Omit r for ascending)
- <FILECOUNTER> ['find . -type f | wc -l'] ~ Counts number of files in current directory
- <DEEPDELETION> [find . -name <FILENAME> -type f -delete] ~ Deletes specified file in directory + subdirectories

- xdg-open . (Opens the directory in default viewer)
- grep [options] [PATTERN] [FILE] ~ REGEX essentially  ##GREP
  - [COMMAND] | grep -B/F{N} [PATTERN] -> Show only lines of output containing [PATTERN], as well as {N} lines {B}efore or {A}fter
  - Full stop treated as regex character meaning "anything", e.g.: grep 1.195 will match 10195, 14195 1-195
    - To treat as literal: * Escape character, *e.g.: grep 1\\.195*
                           * Pass -F option,   *e.g.: *grep -F 1.195*

# Installing cmake/ NiftyFit
- sudo yum -y install cmake [OR] download files and ./bootstap in relevant directory
- Download code from https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit-Release
  - mkdir build path
  - cd build (build path in documentation)
  - c++11 compiler option support required, need to change [CMakeLists.txt]
    - Open file, find {CMAKE_CXX_FLAGS} -fopenm and ADD [-std=c++11] option
    - sudo make (then sudo make install (? Not sure if absolutely necessary))
    - Try command: e.g. fit_asl -h

**[Registration Notes]** ##Registration ##Compose
# MPM registration to MNI space
- Was doing AFFINE registration of TS to MNI space
  - Then AFFINE registration of MPM to TS_MNI

- Correct approach : 1. RIGIDLY register MPM to TS => [TRANS1]
                      **Note** *Can (should) use reg_aladin again, AFFINELY, with [-lp] & [-ln] options*
                     2. AFFINELY register TS to MNI => [TRANS2]
                     3. Use reg_transform to compose transforms => [TRANS3]
                     4. Use reg_resample to resample MPM to MNI space using TRANS3

**[Cluster notes]** ##CLUSTER
# Login: ssh pborges@comic2.cs.ucl.ac.uk
# pword: hGh48FHn

[Cluster website] hpc.cs.ucl.ac.uk
>                 Login: hpc
>                 PWD: comic

[Mounting locally] mcomic  *Check alias in ~/.zshrc*

[File transfer w/out VPN] 1. ssh -L 1234:comic2.cs.ucl.ac.uk:22 pborges@storm.cs.ucl.ac.uk *Establish connection in one terminal* ##NOVPN
                          2. scp -P 1234 pborges@127.0.0.1:CLUSTERFILE . *Copy file from cluster to local machine*
                          3. scp -P 1234 -r LOCALFILE pborges@127.0.0.1:CLUSTERDIR *Copy file from local machine to cluster* *ORDER MATTERS*

[Outside access] * Thinlinc OR bouncer

[Copying] * scp -r FILE pborges@comic2.cs.ucl.ac.uk:~/ *TO cluster*
          * scp pborges@comic2.cs.ucl.ac.uk:~/FILE . *FROM cluster*

[Running scripts] * qsub SCRIPT.sh
[+ job status]    * qstat *STATUS of script*
                  * qstat -j ID *STATUS of specific script*
                  * qstat -r *RESOURCE requests of jobs*
                  * qalter -w p ID *Check queueing status*
                  * qalter -p # ID *Alter job priority: -1024:1024*
                  * qconf -sh *Node listing* ##NODES
                  * qhost -F gpu, gfx | grep -B1 'Host Resource' *Check UNALLOCATED GPU resources* ##GREP
                  * watch qstat *Shows qstat output, updates every few seconds*
                  * tail -f JOBID *Shows contents of output files as they run: Job needs to be in 'r' state, obviously*
                    - e.g.: * tail -f MYSUB.sh*: Error + output
                            * tail -f MYSUB.sh.o*: Only output
                  * qstat -s {p|r|s|z|hu|ho|hs|hd|hj|ha|h|a}[+] *Show only jobs in specified state*
                  * qstat -u USERNAME | wc -l *Counts jobs in queue for that user*

[Multicore] * #$ -R y
            * #$ -pe smp 8 *Not necessarily useful*

[Deleting scripts] * qdel + jobID

[Node exclusion] * (qsub) -l h=!NODE1[&]!NODE2 {Cheech}{Allen}
[Node listing] * qconf -sh

[Test examples] * qrsh MEM_VARS == 14Gb *shemp == test*

[Exiting] * Exit

[Useful commands]
- qsub <scriptfile> *submit to cluster*
- qstat - j ID *Check STATUS*
- -l tmem=xG -l h_vmem=xG -l h_rt=x *Memory/ time options*
- scp (-r) <FILES> pborges@comic2.cs.ucl.ac.uk:~/<DIR> *Copy TO cluster*
- scp pborges@comic2.cs.ucl.ac.uk:~/<FILE> . *Copy FROM cluster*

[Read further] * Sungrid Engine (SGE): http://gridscheduler.sourceforge.net/htmlman/manuals.html


<FURTHER NOTES: GPU JOBS> ##GPUJOBS # 20.11.18
[Submission scripts] * #$ -P gpu *Stipulate that it's a GPU job*
                     * #$ -l gpu=1 *Only ONE GPU per job*
                     * #$ -l tmem=11.5G *No need to set vmem*


ssh college machine first tails.cs.ucl.ac.uk globally accessible
thinlinc : www.cs.ucl.ac.uk/csrw Installation

request project storage space ~ 50gb (stored here)
comic as login node: cluster status
/share/apps software installed here: Need to call/ add to path
/examples: Examples of how to run etc. Wrap code in exectution script
Interactive session request: Test nodes
Overestimate time: Say, 4x
-j y: One output err file
qsub script

# Stats on job(s) running
qstat
qstat -u '*'

# Deleting a job
qdel + jobID
qstat -j ID

#$ -R y
#$ -pe smp 8 Not necessarily useful (more efficient for one)
scales with memory requests
nslots == num

array jobs for multiple iterations: See examples
source /share/apps/whatever to set up environment properly

# Running scripts on test nodes- So do not need to wait to check for errors
qrsh -l <mem_vars> set to 14Gb [shemp == test]
make sure to log out cleanly [exit]

sungrid engine

singularity solution
/scratch0


**[Shell scripting notes]** ##Bash ##Shell
- Can include commands in a shell script that will be executed by the cluster when called using qsub
- Adapted my POSSUM scripts to do this by: 1. Adding memory options
                                           2. Adapting file structure to that in cluster storage
                                           3. Moving scripts to cluster storage

# If statements:                           # For loops:
if <CONDITION>; then                       for <VAR> in <DIR>; do
  [DO STUFF]                                 [DO STUFF]
else                                       done
  [OTHER STUFF]
fi

# Variables:
- Set command to variable: [VARIABLE] = "$([COMMAND])"
- Pass variables onto script by adding them after calling
  - e.g. <SCRIPT> [OPTION1] [OPTION2] [OPTION3]
  - In script, these will correspond to [$1] [$2] [$3] etc.
- Can access parts of string via: ${STRING:START:#CHARS}  *START IS FIRST CHARACTER IF ZERO INDEXING*  ##STRINGINDEXING
  - {STRING:0:#END} *Returns string without final #END characters*

# Find + Editing:
find [LOCATION] -type f -mmin +[TIME] ! -name '*[REGEX]*'
$(echo "[FILENAME]" | sed 's/./[CHAR]/[CHAR_NUM]')
  See also: https://stackoverflow.com/questions/4014074/how-to-read-output-of-sed-into-a-variable ##SED
echo "[STRING]" >> [FILENAME]


**[Vim notes]** ##VIM
- Create .vimrc in ~/
- Add [set backspace=2] (Makes backspace work as expected)
  - Also add [stty erase '^?']

# Replacing  ##VIMREPL
- [:START,STOP s/search/replace/g] *Use %s to look in whole file*
- set mouse=a                      *Allows use of mouse to place cursor*
- inoremap <C-g> <C-x><C-f>        *Allows for directory completion, this is remapping*
- [:g/<SEARCH>/ s/<PATTERN>/<REPLACEMENT>/] *Selectively carry out pattern replacement in Vim, i.e.: Only replace in those lines that contain <SEARCH>*
  - See: https://stackoverflow.com/questions/58096643/replace-string-only-in-matched-search-in-vim/

# KCL bouncer
ssh pbo18@bouncer.isd.kcl.ac.uk  *Bouncer*
ssh pedro@lihe041-u-pc.isd.kcl.ac.uk  *Desktop*

# Counting
- [:START,STOP s/pattern//gn]  *Use %s to look in whole file*
- [:START,STOP ///gn]          *Count number of occurrences of last pattern*
- Omit g to display number of lines where pattern matches

# Advanced
[Quotes]
- Grouping quotes <"">: Signals that contents to be interpreted literally (e.g.: spaces)
                        Prevents aliases/ wildcards from being expanded
- Single quotes <''>: Completely literal, including backslashes
                      Can combine with GQ to allow for variable expansion: *e.g.: '"${my_var}"'*
Reference: https://stackoverflow.com/questions/13799789/expansion-of-variable-inside-single-quotes-in-a-command-in-bash

[Delimeters] By default should use forward slash
             BUT can be problematic for regex involving directories: In this case can use any special character, provided same syntax is followed
              *E.g.: %s!SEARCH_FOR_THIS!REPLACE_WITH_THIS!g* Used exclamation point as delimeter in this case
             References: https://stackoverflow.com/questions/1684628/in-vi-search-and-replace (Eric's answer)
                         https://stackoverflow.com/questions/6714469/search-and-replace-in-vim-results-in-trailing-characters (Realisation of error)

[Command Line operations]
- vim <FILENAME> -c 'OPERATIONS | SPLIT | BY | VERTICAL | BARS'   *Useful to call vim in bash. Typically would end with | wq*


**[UBUNTU]**
[Useful add-ons] - Workspaces: https://github.com/zakkak/workspace-grid/blob/3.30/Readme.md
                 - zsh + Oh My Zsh
                 - Terminator terminal
                 - Pycharm (Coding + debugging in python)
                 - _WIP_


**[DGX-1]**
# Connecting: [sudo openvpn --config "/home/pedro/Downloads/kcl.ovpn"] > [ssh dgx1]
- Allows for use of up to 8 GPUs for jobs
- Run everything in **DOCKER CONTAINERS**  ##DOCKER

[Storage] * Data + models: </raid/pedro>
          * Outputs + NiftyNet: <home/pedro>


[DOCKER] * bash build.sh *Refreshes images*
         * bash run.sh   *Creates new container*

         * docker images (-a)
         * docker attach <ContainerName>
         * docker ps *-a if want to peruse dead containers*
           * docker start <DeadContainer>
         * docker rename <CurrentName> <NewName>
         * CTRL-P CTRL-Q *Exit curent docker container*

[TMUX] Cheatsheet: https://gist.github.com/henrik/1967800  ##TMUX
       * Allows for work to be done in parallel in a docker container
       * tmux                          *Start new session*
       * ctrl+b c                      *Create new tmux "pane"*
       * ctrl+b :kill-session          *Kill tmux session* (Kills ALL panes)
       * ctrl+b x                      *Kill current tmux pane* (Prompt before kill)
       * ctrl+b :kill-pane -t <NUMBER> *Kill specified pane* (No prompt)
       * ctrl+b [                      *Allow scrolling functionality, follow with q to quit*
       * ctrl+b d                      *Detach from tmux session*
       * tmux ls                       *List tmux sessions*
       * tmux attach -d -t <NUMBER>    *Re-attach to specified tmux session*


# qform vs sform ##QFORM ##SFORM
Q19. Why does NIfTI-1 allow for two coordinate systems (the qform and sform)? (Mark Jenkinson)
The basic idea behind having two coordinate systems is to allow the image to store information about (1) the scanner coordinate system used in the acquisition of the volume (in the qform) and (2) the relationship to a standard coordinate system - e.g. MNI coordinates (in the sform).

The qform allows orientation information to be kept for alignment purposes without losing volumetric information, since the qform only stores a rigid-body transformation which preserves volume. On the other hand, the sform stores a general affine transformation which can map the image coordinates into a standard coordinate system, like Talairach or MNI, without the need to resample the image.

By having both coordinate systems, it is possible to keep the original data (without resampling), along with information on how it was acquired (qform) and how it relates to other images via a standard space (sform). This ability is advantageous for many analysis pipelines, and has previously required storing additional files along with the image files. By using NIfTI-1 this extra information can be kept in the image files themselves.

Note: the qform and sform also store information on whether the coordinate system is left-handed or right-handed (see Q15) and so when both are set they must be consistent, otherwise the handedness of the coordinate system (often used to distinguish left-right order) is unknown and the results of applying operations to such an image are unspecified.


# Notes on possum.cc
<Functions/ Creating variables:>

- TYPE name: Create an instance of NAME with the given TYPE
- read_volume4DROI(output_name, input_volume...)
- <NEWMAT> Indices start at ONE
- <NEWIMAGE> Indices start at ZERO
- <FUNCTIONS> Found in possumfns.cc (Can find in dev folder)

Tissues changes:
Existing code:

- Matrix tissue; [tissue characteristics: T1,T2,PD,]
                 [ChemicalShift = value(ppm) * gammabar * B0]
- <Need to change this to allow for 3D T1, T2, PD maps to be allowed as inputs>
- <3D matrices as inputs: One for each of the variables>
-
- tissue = read_ascii_matrix(opt_tissue.value()); [in SI already]
- cout << "T1,T2,SD,CS:"<< endl;
- cout << tissue << endl;


# ZSH Notes ##ZSH

**ERRORS**
- If error "Corrupt zsh_history" then: cd ~
                                       mv .zsh_history .zsh_history_bad
                                       strings .zsh_history_bad > .zsh_history
                                       fc -R .zsh_history

- Return to bash, then back to zsh (bash -> zsh in terminal) *For Tensorflow missing related errors*
- **/<STRING>* *Find files using STRING in subdirectories: useful for copying*

# Running possum as test

- ./runPossum.sh
- To select desired slice, pass --zstart=<PARAM>, where <PARAM> is in mm (So ~0.09 for middle slice)
- Need to change other params for entire brain simulation:
- --numslc [Number of slices] --nx [x resolution] --ny [y resolution] --gap [Btwn slices in m] --te [In s] \
  --dx [x dim voxel m] --dy [x dim voxel m]


# Issues

- Multiple libraries (.h files) not found in possumdev dir
  [SOLVED]: Changed make file appropriately, added full directories to include search path

- Unsure how to create vectors
  [SOLVED]: Use of NEWMAT class RowVector, see <NEWMAT> possum_notes section

- Unsure how to find where certain functions/ classes are defined (e.g. volume4D)
  [SOLVED]: Use of NEWIMAGE class volume4D, see <NEWIMAGE> possum_notes section ~ https://users.fmrib.ox.ac.uk/~mark/newimage/newimage.html

- Compulsory argument -e, -mainmatx: Where to find example data for cmdline execution of possum
  [SOLVED]: Have to run POSSUM_MATRIX function to create mainmatx for use in POSSUM

- Bizarre output from POSSUM_WORKING (using MPMs)
  <UNSOLVED> ~ Perhaps not looking at correct slice

- MNI MPM output does not match normal POSSUM output
  <UNSOLVED> ~ Things to try: - [DONE] Check for instances of "tissue" in code, compare if mpm_maps is doing same
                              - [DONE] Changed indexing: mpm_maps uses NEWIMAGE (zero index.) BUT tissue uses NEWMAT (one index.)
                              - [DONE] Validate using possum_working ~ Will ascertain if problem in functions or PW implementation
                                - *Found to be exactly the same~ Suggests problem is in how voxels are calculated*
                              - Most differences seem to be in grey matter: Double check values `23.05`
                                - *Checking coord (68, 108, 172)* => *Tissue probs do NOT add up to one*
                                - *Means that den calculations are off: Was assuming sum to one*
                                - *Explains perceived higher intensity values*
                                - **Incorrect, does not matter that sum isn't one, this is accounted for already**
                                - Check if weighting den by sum gives equal output
                                  - *Created adjusted 'ones' phantom that contain sums of segs (3D + 4D)*
                                  - **Incorrect, does not matter that sum isn't one, this is accounted for already**
                              - Check intermediate outputs?
                              - Check voxel calculations: See if inputs to "voxel" class are the same (e.g.: den)

# To do List
- Run possum_working and check for similar functionality to original possum [DONE]
- Run possum_working using MPMs and check output [DONE]
- Try to obtain reasonable output using MPM ~ Need to find appropriate zstart (i.e. middle slice) for best visualisation [DONE]

- Bulk assignment technique check: [DONE]
  - Create MPM for MNI brain (Using tissue segmentations) [DONE]
  - Pass as inputs to POSSUM_WORKING [DONE]
  - Compare this output to normal POSSUM output: Should be exactly the same [DONE]~ Is not, see **Sequence section/expo_plots.png**
  - Set all param values of tissues to be the same (i.e. all T1 the same, all T2s the same) as further validation
  - Plot exponentials to check how good approximation is vs (realistic) compound exponentials
    - *Need to be careful with how T1, T2s are added, not simple addition*
    - See voxel [113, 110, 172]~ Purely WM BUT fractional i.e. != 1
      - (Real) Signal value in MPM: 2.7435994217612073236e-11
      - (Real) Signal value in norm: Different, see **Sequence section/expo_plots.png**

- Look into how POSSUM utilises params to create Pulse Sequences (pulse.cc, possum.cc to check) <NEXT>
- Look into MT maps (susceptibility) -> How POSSUM uses it [DONE]~ Does not account for them in any way currently
- Correct for erroneous output <NEXT>

# Questions

- Bloch equation in voxel functions: Where?
  - <CURRENT> Will continue in next meeting
- Why difference exists between outputs of MNI (MPM vs normal)
  - <CURRENT> Probably due to how exponentials are dealt with~ Currently being checked
- Why zz increases when numslc = 1?
  - [SOLVED] POSSUM looks at adjacent slices because of non-perfect excitation pulse





# Log: 24.05.18 + 25.05.18 ##EXPO_INIT

- Altered artifical MNI MPM creator code to *APPROPRIATELY* account for relaxation constants
  - *IMPORTANT* MUST use affine (i.e. saveimg(mat, AFFINE)) of segmentations when saving images
  - Differences still seen in outputs: Suggests correlation between params, more complex than addition between them

- Created more simple validation test where param values for ALL tissues were set to be the same -> Should fix output diff. in theory
  - Small differences found when comparing outputs, less so than with previous test
  - There is a time dependence in PV voxels: Cannot approximate compound exponential with an exponential (e1 + e2 != exponential)
    - BUT should be able to approximate in case where params for tissue are equal

- Moved on to compare intermediate values when running POSSUM:
  - Look at (Real) Signal value [sreal] of voxel [113, 110, 172] -> Has partial volume effects so suitable for checks
  - Changed both runPossums to look at relevant slice (zstart=0.087):
  - (Real) Signal value in MPM: *2.7435994217612073236e-11* 432047 voxels, zstart=171, zend=172
  - (Real) Signal value in norm: *8.783293742982092283e-09* 302548 voxels, zstart=171, zend=172 ~ Fewer voxs. because zeros excluded

- Compared MPM_MNI to MPM_MNI_corr (expo corrected): Very similar BUT PD different for some reason (double in MPM_MNI)
  - Re-ran MPM creation code again, rectified issue





# Log: 26.05.18

- Simulated multiple (25) brain slices using artificial MPM & normal tissue segmentations





# Log: 29.05.18

Aims:
- Need to double check why "simple" maps don't seem to be working (i.e. when all tissues set to same param values) [DONE]
- Compare multiple slice simulations <NOT-DONE>
- Email mark about magnetisation transfer maps and usefulness [DONE]
- Visualise Pulse sequences [DONE]

Notes:
- Simple maps analysis: Extremely similar for most part to 1/100
  - validation.py script written to compare outputs, biggest differences in CSF around brain
  - Differences arise where there are PV effects s.t. sum(segs) != 1
  - This means that the T_approx solution becomes time dependant, and therefore cannot be estimated
    - This is the **same** reason why differences arise in the other non-simple MPM simulations: Time dependence

- Began process of visualising pulse sequences
  - Chose simple EPI sequence file as starting point
  - Created simple script for visualisation of gradients + ADC: Sequence section/POSSUM_sequences.m
  - Created simple script for visualisation of slice profile:  Sequence section/sequences.py *Also includes "Bad" sequence figures*
  - **Saved images**:  Sequence section/expo_plots.png + Sequence section/profile.png

- MT Maps: No easy way to incorporate into POSSUM right now





# Log: 30.05.18

Aims:
- Visualise variations of EPI sequence [DONE]
- Visualise GE sequence + variations [DONE]
- Simulate simple EPI image using my MPMs

Notes:
- Generation of GE sequence:
  - Acquired GE params from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2753785/
  - Params: [--zstart=0.09 --slcthk=0.002 --dx=0.002 --dy=0.002 --nx=128 --ny=128]
            [--te=0.05 --tr=0.0273 --gap=0.001 --numslc=1 --bw=100000]
  - *Sequence diagram seems off, need to double check*

- Generation of further EPI sequence:
  - Acquired EPI params from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2798023/
  - Params: [--zstart=0.09 --slcthk=0.002 --dx=0.002 --dy=0.002 --nx=128 --ny=128]
            [--te=0.07 --tr=5.4273 --gap=0.001 --numslc=1 --bw=441000]
  - **Saved images**: Sequence section/epi_sequence.png + Sequence section/epi_sequence2.png

- Simulation using REAL MPMs:
  - TBD

# Weekly meeting notes: MRes plan (30.05.18)
- [Step 1] **SANITY CHECK** Compare outputs of normal tissue segmentation based (TSB) POSSUM
                            and new MPM based POSSUM (MPMB):

  - Validate using BrainWeb Phantom: Use it for TSB sim. + create artificial MPMs for MPMB sim.
  - Tried two approaches: Simple (Bulk asssignment) + Standard (Weighted expo. sum)
    - In standard: PV voxels display different results because of impossibility of
                   approximating sum of exponentials with single exponential
    - In simple: Non-full PV voxels exhibit differences: Due to time dependence
    - *Need to try to find ideal time for approximation: NOT just use one in solving equations*

- [Step 2] **METHOD COMPARISON** Compare performance (in terms of output) of *STATIC
                                                                             *MPM BLOCH EQUATION (NEW POSSUM)
                                                                             *SEGMENTATION (OLD POSSUM)

  - <STATIC>: Simple simulation model based on using exponentials instead of solving full BEs
    - Cannot account for complex sequences, MT, fat, anomalies (e.g. tumours) easily
    - Based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4532609/pdf/nihms692268.pdf
    - <Paper: MR Image Synthesis by Contrast Learning On Neighborhood Ensembles>
    - *Needs to be implemented by me: See paper*
  - <MPM BLOCH EQUATION>: Self explanatory.
  - <SEGMENTATION>: Selft explanatory.
  - Want to use these THREE methods to simulate images of simple pulse sequences (e.g. T2)
    - Compare simulations to output: Which methods performs best?
    - **

- [Step 3] **NEURAL NET VALIDATION** Train NN to learn between two specific modalities: *T2_TR=50
                                                                                        *T2_TR=70

  - Learn mapping between two SIMULATED modalities
  - Use learned mapping to map between REAL images
    - *EXTRA milestone*

[Other notes]
- Try to calculate ideal time (te maybe?) for solving exponentials
- Read paper to implement STATIC method
  - Fat suppression important: POSSUM can take this into account, but STATIC method cannot
    - Need maps: Fat/ water Dixon acquisition
  - Same with MT, but POSSUM cannot take this into account for now
- Need to try to perform simulations using my MPMs
- Would be good to develop more complex sequences in POSSUM





# Log: 04.06.18

- Created new MNI based MPMs using TR instead of defaulting to solving exponential for t = 1 (as was being done before)
- Ran POSSUM for these maps: Results still seem very similar to before, same problems with PV voxels
  - Checked exponentials: GT & new exponentials cross at TR as expected
  - *Will try to set up for loop for multiple TRs to see how it compares*
  - Tried TR = 3.1 (Standard sequence acquisition) for above
  - Trying TR = 0.12 (== TRslc): **Worked**, exactly the same as GT TS POSSUM
    - Seems like measurements taken at t = TRslc

[FILES]
- Validation (TR considerations): * 64 x 64 outputs {From standard sequence}
                                  * TR = 3.1s (= TR)
- Validation (TR considerations HR): * 128 x 128 outputs {From standard sequence}
                                     * TR = 3.1s (=TR)
- Inputs: * MNI MPMs with expo. solved for t = 3.1s

# MR Image Synthesis by Contrast Learning On Neighbourhood Ensembles: ##JOGPAPER

- Simulate MR sequences using *simple* exponential based equations solving (instead of full BEs)
- Estimate the pulse sequence params. used to generate new atlas images that have the same contrast as subject image
- Nonlinear regression intensity mapping trained from new atlas to target atlas THEN applied to subject space to yield particular pulse sequence in atlas

[Introduction]
- Problem with MR: Need to make decision about what pulse sequence to choose
  - Expensive to acquire longer scans (e.g. T2-w)
  - Inhomogeneous quality of data because of requirements of sequences, e.g.:
  - DSE: Low resolution
  - T1-w: High resolution
  - FLAIR: SD Artifacts
- Want to standardise intensity of data + improving utility of MM data via image synthesis

- Image synthesis: Learning + applying intensity transform. to produce images that perform better in proc. tasks
  - Produce synthetic images: Any pulse sequence

- Intensity standardisation: No consistent image intensity scale for diff. tissues: Not quantitative
  - No problem for diagnostics
  - Problem for image processing algorithms
  - Difficult to compare MR data from different sites/ even intra-subject scans

- Better to register to SYNTHETIC image (created from different modality) than to directly register, e.g.:
  - T2-w -> T2-w (SYN from T1-w) [GOOD]
  - T2-w -> T1-w (DIR) <WORSE>
  - Already been done for FLAIR-less datasets

- Previous work: Regression approach
  - Random forest regression for synthesis of MR images
  - Patch based: Voxel region A -> voxel region B
  - CONT: Use MR physics + Atlas info.
    - PD, T2, T1, GAIN, TR, TE
    - Estimate pulse params. using JUST image intensities
    - Use PS + MR params. to synthesise new subject image
    - Learn nonlinear regession btwn SYNTH & desired atlas contrast: Apply to subject image to synthesise new image with desired contrast

[METHODOLOGY]
- <B>: Subject image set (multiple PS)
  - e.g.: MPRAGE, SPGR (SPoiled GRadient echo), DSE etc.
- <A>: Atlas collection (contrasts C, multiple PS)
  - **meaning** contrast == tissue contrast provided
              - e.g.: MPRAGE == T1-w contrast
  - Also contains quant. maps: Use to synth. subject image as if it had been imaged with <A> PS.
- *No need for intersection in PS btwn A and B*

- Algorithm steps: 1. Estimate PS params from <B>_i.
                   2. Generate <A>_bi ~ Atlas image with same contrast as <B>_i
                   3. From expanded <A> learn nonlinear transform btwn contrasts (C_i and C_r)
                   4. Apply transform to <B>_i, creates <B>_r of desired contrast.

- Estimating PS params.
  - Intensity observed at given voxel assumed to be due to tissue params: *PD, T2, T1, GAIN, TR, TE* ~ [BETA]
  - Also due to PS (+ its imaging params.) ~ [THETA]
  - Pulse sequence example:
            B(x) = G * PD * DECAY_T1 * DECAY_T2
  - Mean tissue intensity equations:
    - <B>: CSF_ave = f([BETA]_CSF; [THETA])
    - <B>: GM_ave = f([BETA]_GM; [THETA])
    - <B>: WM_ave = f([BETA]_CSF; [THETA])
- [THETA] is unknown: Four params: TR, TE, G, Fl Ang. etc  {PS dependant}
- Three equations, three/ four unknowns: Newton's method
- Use approximations to connect intensity to NMR params.
  - Exponentials described in Eq. 2 - 4
  - Do NOT account for many scanner params.
  - Not an issue, approximation is good enough





# Log: 05.06.18

- Validated TR == TRslc by creating new images with different sequence
  - --dx=0.002 --dy=0.002 --nx=128 --ny=128 --te=0.07 --tr=5.4273 --gap=0.001 --trslc=0.50 --numslc=1 --bw=450000*
  - *NOT* the same, see below.
- [NOTE] In an EPI sequence entire slice acquired with SINGLE excitation
  - Therefore TRslc == total elapsed time
  - BUT given that multiple slices are processed in POSSUM to simulate one slice, may not be 100% accurate

- Retried approach using above sequence BUT with MPMs calculated using TR = 0.12 (as before)
  - **WORKED** the same, minor intensity differences
    - *Need to figure out exactly what 't' needs to be chosen to ensure valid validation*

- Started to implement approach in: https://www.ncbi.nlm.nih.gov/pubmed/26072167 ##JOG
  - Implemented, in addition, simple SE intensity equation found in: https://www.cis.rit.edu/htbooks/mri/chap-10/chap-10.htm
  - <N> Scanner GAIN: How does POSSUM account for it?
  - <N> [LOW RES] vs [HIGH RES] output: How to compare POSSUM and SE?
  - <N> Multiple slice calculations in POSSUM: How to simulate using SE approach

[Other notes]
- Emailed Ivana to update her on the project
- <N> Need to text Mark to choose appropriate meeting time for next week

[FILES]
- Validation (TR considerations correct 12): * 128 x 128 output
                                             * Correct POSSUM output (using 't' == TRslc)
- Validation (TR considerations correct 5012): * 128 x 128 output
                                               * Outputs by using new PS (See above in Log)
                                               * TR_50: GT POSSUM
                                               * MNI_TR50: Uses MPMs from 't'=0.50s <INCORRECT>
                                               * MNI_TR5012: Uses MPMs from 't'=0.12s [CORRECT]
- Inputs: + MPM_MNI_corr_newest_TR50.nii
          * MPMs created by solving with 't' = 0.50





# Log: 06.06.18
- Created Higher resolution images using EPI sequence:
  - --dx=0.001 --dy=0.001 --nx=256 --ny=256 --te=0.2 --tr=2.0 --trslc=2.0 --gap=0.001 --numslc=1 --bw=2000000
  - Paper: http://www.pnas.org/content/pnas/89/12/5675.full.pdf

- Attempting to use Pedro's MPMs to simulate SHR SE image
- Needed to resample MPMs into 'standard' space (non-rot): 1. Register one PM (PD was used) to *MNI brain* (space of segmentations)
                                                              using reg_aladin (i.e. affine)
                                                              **Register to brain instead to get needed POSSUM orientation**
                                                           2. Resample PM into space of *MNI brain* using reg_resample (linear)
                                                           3. Resample all other PMs into space of *MNI brain* using same transform
                                                           4. Assemble all PMs into properly resampled MPM, some variants:
                                                             - Flipped along second axis (To rectify orientation) [MPM_Pedro_flip.nii]
                                                             - Removed infinites [MPM_Pedro_ninf.nii]
                                                             - Scaled all values above TH to TH (200 chosen initially) [MPM_Pedro_smol.nii]
                                                             - Removed nans (Now standard part of procedure)
                                                             - <See ...Project\MPM section\mpm_registration.py>
- Ran POSSUM with flipped MPM `MPM_Pedro_flip.nii`
  - *All zeros*
    - Things to check: * Register MPMs to space of MNI brain: Should conclusively correct for orientation errors <WRONG>
                       * Set all absurdly large values to zero [DONE]~ 1. Capped T1 at 10
                                                                       2. Capped T2s at 1
                                                                       3. Removed all negatives
                       * Normalise PD: Currently values range from 10 ~ 150 [DONE]~ Normalised wrt maximum

[FILES]
- SHR POSSUM outputs: * 256 x 256 outputs
                      - First (Bad params)
                        * HR_TR50: GT POSSUM
                        * See above for sequence
                      - Proper params
                        * SHR2: GT POSSUM
                        * Same sequence but TE = 0.15s, TR = 2.0s, bw=1000000
- Inputs: + MPM_Pedro.nii
          + MPM_Pedro_flip.nii
          + MPM_Pedro_ninf.nii
          + MPM_Pedro_smol.nii

# Meeting notes: Next steps
- Compare MPMs to bulk tissue assignment
  - Look at tissue contrast (e.g. White vs Grey matter) against TR
  - For static, Full Bloch Equation simulator
  - Register MPMs to MPRAGE segmentations (Jorge doing the segmentation): Use Affine





# Log: 06.06.18 + 07.06.18

- Obtaining reasonable output from POSSUM given slight changes to MPMs
  - Re-ran POSSUM w/ corrections: Obtained unexpected outputs </home/fsluser/simdirIrinaTest/SHR POSSUM Pedro outputs/z009>
    - *Actually this was a sagittal reconstruction* (Though it was the base of the neck originally)

- Rectified orientation by registering MPMs to MNI brain instead of MPRAGE images
  - Re-ran POSSUM w/ these for 128 x 128 output: Seemed reasonable </home/fsluser/simdirIrinaTest/SHR POSSUM Pedro outputs/z009 128>

  - Re-ran POSSUM w/ these for 256 x 256 output and [MPRAGE] sequence: Seemed good, but had some artefacts, notably in CSF of ventricles
                                                 </home/fsluser/simdirIrinaTest/SHR POSSUM Pedro outputs/z009 256 MPRAGE BAF>
                                                 *Likely caused by low exclusion limit of T1/T2s*

- Slightly altered the MPM registration/ creation script: * Originally was excluding T1 > 10s, T2* > 1s
                                                          * Decided to increase limit to 50s for both [MPM_Pedro_MNI_abs_HT.nii]
                                                          * Decided to have version with no limit [MPM_Pedro_MNI_abs_NC.nii]
                                                            * abs denotes that the absolute value is taken (not done before)

- Re-ran POSSUM w/ no correction map: No artefacts present and no absurd values detected
                                      </home/fsluser/simdirIrinaTest/SHR POSSUM Pedro outputs/z009 256 MPRAGE NC>
                                      **Suggests this is ideal map choice**

[TO DO LIST]
- Compare outputs of static vs MPM BE
- Re-register maps: Assumed all PMs in same space, so only one registration required BUT this may not be the case
- Attempt other sequences





# Log: 11.06.18

Aims: - Segment brain: Will allow for exclusion of voxels outside cranial region == more efficient
      - Find appropriate sequence
      - Find way to extract fat/ water maps from Dixon acquisition
        - <C:\Users\pedro\Desktop\Project\MyScans\20180207_bor_pe_NIFTY\0022-t2_tse_dixon_tra_192_2mm_W.nii>
        - Can attain granularity of other PMs or have to resort to TB value?


[COMMANDS]
- bet <input> <output>: Crops image to just brain
- fast <input>: * Segments (cropped) image to different tissue types (CSF, GM, WM)
                * Order:                  [0]CSF, [1]GM, [2]WM
                * Order needed by POSSUM: [0]GM,  [1]WM, [2]CSF

*NOTE*: * Fast produces BOTH prob and pve files
        * prob == tissue probabilities
        * pve == tissue proportions
        * VERY similar BUT **PVE should be more accurate**: Uses PV vector + spat. regularisation


- Cropped MPRAGE_MNI.nii (MPRAGE image registered to mni brain) using BET + segmented using FAST
    [MPRAGE_crop.nii.gz] + [MPRAGE_crop_pve_012] ~ POSSUM inputs
  - Loaded into python to concatenate in correct order (a la brain.nii)
    [MPRAGE_segs_poss.nii.gz] ~ POSSUM inputs
  - Ran standard POSSUM w/ these tissue segmentations

- Fat Dixon acquisition: Took as is and scaled (because arbitrary units)
  - Added to MPM as 4th volume:
  - <C:\Users\pedro\Desktop\Project\MPM section\fat_to_mpm.py>

- Ran standard GE sequence simulation for one slice using more robust map
    [imageMNI_SHR_Pedro_fat_abs.nii.gz]
  - Artefacts aplenty: Surrounding of brain seems corrupted

- Changed POSSUM_WORKING:
  - Needed to add ability to consider CS maps
  - While will always have T1, T2* and PD, CS seems more "optional", therefore not originally implmented
  - Added a check to see if CS map has been passed by looking at size of 4th dimension
    - If == 4, then set value of "props" (Row Vector denoting param values) to corresponding map value
    - Otherwise keep at zero

[TO DO LIST]
- Check registration logic: MPRAGE -> MNI vs MPMs -> MNI
  - These do NOT align properly (Consider asking Jorge tomorrow)
  - Consider affine (instead of just rigid) registration to MNI brain
    **OR** register MPRAGE *THEN* register MPMs to resampled MPRAGE





# Log: 12.06.18

- Re-made the MPMs based on yesterday's thoughts: 1. Registered MPRAGE to MNI brain *RIGIDLY*
                                                  2. Register PMs to MPRAGE *AFFINELY*
                                                 (2a. Create CS map using Fat Dix. in process)
                                                  3. Concatenate into **NEW** MPMs

- Changed <runPossumWorking_MPRAGE.sh> to use the new MPMs
  - Ran using standard SPGE sequence w/ TR = 2.0, TE = 0.002, theta = 8
  - Obtained reasonable looking output
    - Can see detail in GM + WM that is otherwise not there when using "flat" tissue approach

- Changed the Jog simulator code: 1. Changed params to match that of acquisition outlined above
                                  2. Save output with **IDENTITY** as affine
                                  3. Read in MPRAGE image
                                  4. Register, *RIGIDLY*, output to MPRAGE
                                  5. Output is now in form that can be easily compared

- Changed <jog_sim.py> to having matching params + use new MPMs
  - Ran using standard SPGE sequence w/ TR = 2.0, TE = 0.002, theta = 8 (See above)
  - Obtained reasonable looking output
    - Extremely similar to BE, MPM POSSUM based approach

- Yesterday's tiss. seg. based POSSUM output comparison:
  - Structurally, matches exactly to STATIC + BE MPM
  - BUT **LACKS** intra-tissue detail present in other two
  - Suggests MPM approach is preferable -> Move onto further validation

[FILES]
- Inputs: + MPM_Pedro_MPR_fat_abs_NC.nii.gz ~ MPRAGE (struct.) registered PMs
          + MPM_Pedro_MPR_abs_NC.nii.gz ~ MPRAGE (struct.) registered PMs + CS
          + MPM_Pedro_ninf_MPR.nii.gz ~ MPRAGE (struct.) registered PMs + inf. corr.

- SHR POSSUM Pedro outputs: + z009 256 MPRAGE NEW TR2 NC ~ Sim. using new MPR MPMs
                            + z009 256 MPRAGE NEW TR2 NC Fat ~ Sim. using new MPR MPMs (+ Fat)

# Meeting notes: Next steps
[Short term: Sequence optimisation + PoC stuff]
- Whole brain simulations
- SNR + CNR (for GM + WM) for sequence parameter optimisation (plot isolines, 2d graph)
- Accounting for erroneous value: If greater than (say) 98% percentile then set to mean of surroundings ##ErroneousCorrecting
- Similarity assessment: SSIM (Validation against ground truth: Scans acquired with specific sequence)
[Simulating fat-shift]
- Currently have erroneous output: <MPM_Pedro_MPR_fat_abs_NC>
- Maybe need to get fractional signal btwn fat and water => Use this instead
- *Ask Mark*
[Sequences]
- SPGR: T2 based sequence, less affected by fat shift
- Would be more interesting to simulate T1 based since these are more susceptible to fat shift ##FatShiftSequence
[Future work]
- Train network to, for example, work out segmentation volumes given different acquisitions
  - One subject at first, then multiples
  - **Much later**: Train on histopathological data, incomplete datasets (e.g. missing MPMs) etc.
- Modality independant training: COnversion from one acquisition to another

# Mark meeting notes
- Concern that MPM approach is "not-realistic"
  - Especially when it comes to Fat map: Would perhaps need to simulate macro-molecules
  - Mono-exponential might be an issue, but seems OK so far
- Should try to simulate standard POSSUM w/ Fat shift
- Should try looking at T1 acquisitions
- Can try creating a new sequence (e.g. MPRAGE): Would involve editing a sequence in MatLab (i.e. pulseSeq)
  - Can perhaps automate using function
  - And eventually implement into [pulse.cc]





# Log: 13.06.18

Aims: - Attempt to get fat segmentation from MPRAGE  (4 tissue types)
        - Attempt simulation
      - Look into SNR + CNR: How to calculate + automate
      - Consider doing some erroneous value correction
      - Re-organise file structure: MPM validation, simulation comparisons etc.

[WORK DONE]
- Segmented MPRAGE image with four tissue types to try to extract Fat:
  - Segmentation seems poor: CSF + WM seem present BUT two others seem like a mix of GM
  - </home/fsluser/simdirIrinaTest/Inputs/MPRAGE_crop_pve0123.nii.gz>

- Altered possumfns.cc to change scaling of fat shift: Divided chshift by 1e3
  - Ran simulation, seems odd still (black artefacts in middle of image)
  - *Could check if fat at all present in centre*
  - </home/fsluser/simdirIrinaTest/SHR POSSUM Pedro outputs/z009 256 MPRAGE NEW TR2 NC Fat 1k>
  - Repeated for 1e6 scaling
    - Looks identical to standard non-fat MPM simulation: {Too high}

- Tried running a T1W sequence (runPossumWorking_MPRAGE.sh)
  - Params: TE = 0.004, TR = 0.03, angle = 40
  - Fat shift should be more evident

- <NOTE> **T1 portion of NEW MPM is warped!** ##slanted
         Likely due to AFFINE registration
         Repeated MPM creation using rigid transformation from PMs to MPRAGE_MNI

**FIXED**: 1. Rigidly register MPRAGE to MNI brain (MPRAGE_MNI)
           2. Rigidly register PMs to MPRAGE_MNI
           3. Concatenate PMs into MPM


[FILES]
Inputs: + MPRAGE_crop_pve_0123.nii.gz ~ Tissue segmentations for MPRAGE (struct.) by COMP.
        + MPRAGE_crop_pveseg.nii.gz ~ Tissue segmentations for MPRAGE (struct.) by LABEL

[SEQUENCE NOTES]: http://mriquestions.com/spoiled-gre-parameters.html
- T1-weighted: * Short TE (< 5ms)
               * Short TR (< 300ms)
               * Large angle (30-50 deg)

- T2-weighted: * Long-ish TE (20-50ms)
               * Long TR (>300ms)
               * Small angle (5-15 deg)





# Log: 14.06.18 (Post)

- Started writing code to calculate SNR + CNR
  <C:\Users\pedro\Desktop\Project\MPM section\SNR_CNR.py>
  - Create masks of each tissue segmentation
  - Create an "inverse" mask as well for background noise calculation

[EQUATIONS]
SNR = (Mean intensity value) / (S.D. of noise) : * Mean intensity value of each region of interest (i.e. WM and GM)
                                                 * Noise S.D. found from "negative" segmentation mask
                                                   * i.e. only ones outside brain
                                                   * Work out S.D. here

CNR = (Sa - Sb) / (S.D. of noise) : * Sa, Sb == ROIs (i.e. WM and GM)
                                    * Noise S.D. same as above

[FILES]
MPM Section: + SNR_CNR.py ~ Explained above

[OTHER NOTES]
- Emailed Irina about POSSUM accepting FAT, things to try: * Concatenate normalised Fat acquisition to POSSUM tiss.
                                                             * Ensure in same space
                                                             * Can use <FSLMERGE> for this (instead of relying on python)

- Visited the CS department to gain cluster access for POSSUM: Intro session @ 3pm @ 4.20 Malet Place





# Log: 15.06.18

Aims: - Try running TS POSSUM w/ Fat as additional tissue type
      - Re-run MPM POSSUM w/ fat and check outputs

<NOTE> **Fat MPM was WRONG**: - Mistake 1: Concatenated the FAT, NOT the CS map
                              - Mistake 2: Was taking absolute value AFTER concatenation
                                - Therefore all CS negative values became positive
       - Amended <mpm_registration_one_version.py> by correcting these errors
       - Re-ran T1W sequence w/ MPM POSSUM w/ Fat with corrected MPM
         - Still looks odd
         - *Is there meant to be fat in the centre of the brain?*

<NOTE> **Segmentations and MPMs NOT aligned**: - Mistake 1: Did not rigidly register MPM to MPRAGE_MNI
                                                            Therefore slices did not match between TS & MPM
       - Amended <mpm_registration_one_version.py> by correcting for these errors [+ _res to file names]
       - Re-ran T1W sequence w/ MPM POSSUM for different slices

[FILES]
Inputs: + MPM_Pedro_MPR_fat_abs_NC_rig2.nii.gz ~ Properly created fat MPM
        + tiss_merge_mpr                       ~ 4D tissue segmentation of MPRAGE + Fat-Dixon
        + MRpar_3T, MRar_3T_fat, MRpar_1.5T    ~ Self-explanatory
        + Fat_map_mpr_rig_norm.nii.gz          ~ Fat-Dixon acquisition, normalised

SHR POSSUM Pedro outputs: + z009 256 T1W FAT 1K
                          + z011 256 T1W FAT 1K TISS
                          + z011 256 T1W FAT TISS
                          + z009 256 T1W FAT TISS
                          + z009 256 T1W TISS
                          + z009 256 T1W FAT MPM

[OTHER NOTES]
- Emailed Irina about cluster use w/ POSSUM: Asked about additional resources
  - Have meeting with her after cluster intro session: ~ 4pm in CMIC





# Log: 18.06.18

Aims: - Need to double check that rigid registration has solved mis-alignment problem
      - Run z=0.09 slice to confirm

- Ran MPM POSSUM with new maps:
  - z = 0.09 returns nothing for some reason
  - z = 0.06, 0.07, 0.08, 0.085 work just fine, BUT seem more blurred
  - [W] Attempting to run for z > 0.09 (0.10): Works

[OTHER NOTES]
- Ask Jorge about MPM creation process
  - Specifically, registration/ resampling needed: 1. Rigidly resample MPRAGE to MNI
                                                   2. Rigidly register PMs to MPRAGE_MNI
                                                   3. Join them
                                                   4. Re-register to MPRAGE_MNI





# Log: 19.06.18 - 21.06.18 ~ Cluster week

- Been working on learning how to use the cluster, what follows are notes, rather than
  any specific tasks which might have been accomplished

[Connecting]
CiscoAnyConnect: UCL login (If not on eduroam or using Becket house eduroam)
Windows: Use PuTTY to ssh into cluster
Linux: ssh from normal terminal

[Useful commands]
- qsub <scriptfile> *submit to cluster*
- qstat - j ID *Check STATUS*
- -l tmem=xG -l h_vmem=xG -l h_rt=x *Memory/ time options*
- scp (-r) <FILES> pborges@comic2.cs.ucl.ac.uk:~/<DIR> *Copy TO cluster*
- scp pborges@comic2.cs.ucl.ac.uk:~/<FILE> . *Copy FROM cluster*

[Errors]
- Eqw: * Error queued waiting, use [qstat -j 5087457 | grep error] to check error ##QUEUEERROR ##JOBERROR
       * qmod -cj <JOBID> to clear the error and queue up again *Not sure if this actually resubmits* ##QERROR

- See {Hash}{Hash}Bash
- See {Hash}{Hash}Vim

[Array jobs]
- Allows submission of multiple jobs to cluster by executing a single script
- Uses $SGE_TASK_ID to reference each task, i.e. input.$SGE_TASK_ID, output.$SGE_TASK_ID

- Requirements: 1. [-j y] + [-R] + [-N NAME] options
                2. All input files available in some (referenced) directory


[possumX]~ possumX <subDir> <OPTIONS> **Do NOT call with qsub**

- POSSUM script that automatically submits to the cluster
- <subDir> must include (as it stands): 1. segmented brain
                                        2. pulse files
                                        3. motion file
                                        4. Slice profile
- Main option relates to number of processors: -n <n>: Told to use ~20
- Calls fsl_sub that submits tasks to cluster
  - Includes emailing command -M + -m: Have removed these

[Simulating a whole brain]

              **NOTE** Make sure to run [. /home/igrigore/fsl/etc/fslconf/fsl.sh] before running possumX !!!!

- Note on pulse files: **Independant** of input object (for MPM)
                       -  Could therefore have simple script that, for all TE/ TR, generates all pulse files

<N> possumX does *NOT* run pulse during submission, options:
    - Run pulse beforehand (so possumX has access to these files)
      - Could create automated (maybe array?) script that creates ALL pulse files
    - Edit possumX so pulse IS run as well

<N> Need to change possumX for MPM code
    - Ensure possum_working is called and *NOT* possum
    - Add appropriate options (i.e. -w)
    - Make sure required files are changed *Actually, still technically need input object, i.e. all ones file*
    - Need to check if fsl_sub needs to be changed as well

[Actual Log]
- Simulated one slice (TS POSSUM) using possumX: 1. Created folder w/ required files (SingleSliceTest)
                                                 2. . /home/igrigore/fsl/etc/fslconf/fsl.sh
                                                 3. ~/possumdevpedro/bin/possumX ~/SingleSliceTest/
                                                 4. ~/possumdevpedro/bin/signal2image -p ~/SingleSliceTest/pulse -i ~/SingleSliceTest/diff_proc/signal_proc_0 -o ~/SingleSliceTest/diff_proc/image_cluster_test -a

- Simulated whole brain (TS POSSUM) using possumX: 1. Created folder w/ required files (FullSimTest)
                                                   2. Rest is same as above *BUT* added [-n 20] after <subDir>
                                                     - i.e. [~/possumdevpedro/bin/possumX ~/FullSimTest/ -n 20]

[NO VPN LOGIN]
- ssh pborges@storm.cs.ucl.ac.uk
  - ssh pborges@comic2.cs.ucl.ac.uk

[STORAGE]
- cd /cluster/project0/GIF/Pedro: Extra (~980Gb) storage provided by Jorge
  - df -h <DIR> to check remaining space
  - du -h <DIR> to see breakdown of used space (-h == human readable)
    - du -sh <DIR> to see sum of used space (-s == sum)

[REGISTRATION QUESTION]
- Use affine to register MPRAGE to MNI
- Register affinely one PM to MRAGE_MNI
  - Re-sample the other PMs
  - Can double check that same space by using fslhd: Reading file headers

[FURTHER NOTES ON SNR CNR]
- For noise: Look at region of relative flatness (e.g. white matter) and take noise as S.D. of region
  - Noise in MRI multiplicative: - (MPMs) Will have noise due to scanner + noise due to tissue
                                 - (TS) Will only see noise due to scanner (tissue is flat)
- Run for, e.g. 3 different TR + TE to start off with





# Log: 25.06.18

Aims: - Successfully simulate whole brain in new (Jorge) directory (Using MNI brain)
      - Edit possumX to accept MPMs
        - Try to simulate whole MPM brain
      - Re-edit mpm creation code to adequately resample/ register PMs

Test run TS HR: * [~/possumdevpedro/bin/possumX /cluster/project0/GIF/Pedro -n 181]
                * <DIR>: /cluster/project0/GIF/Pedro/FullSimTest/
                * Outputs stored in <DIR>/diff_proc

Test run TS SR: * [~possumdevpedro/bin/possumX /cluster/project0/GIF/Pedro -n 20]
                * <DIR>: /cluster/project0/GIF/Pedro/SmallSingleSliceTest/
                * Outputs stored in <DIR>/diff_proc

- Created possumX_MPM to allow for MPM POSSUM to be run on cluster
  - Added -w flag, edited -i flag to look for 'ones' instead of brain
- Edited fsl_sub (on cluster) to reduce individual processor memory requirements:
  - From [-q long.q] *8G of mem* to [-l tmem=3G -l h_vmem=3G -l h_rt=36000]
  - Should make task queueing significantly faster (only need to find 4G machine instead of 8G+)





# Log: 26.06.18

Figured out how to auto. run certain commands on login: * **.bashrc**
                                                        * Put all fsl related commands in it so .profile does not need to be called manually

Test run TS SR results: * Pulse was not set up properly: dx, dy == 0.001 *Needed 0.004*
                                                         zstart == 0.09 *Needed 0*
                                                         numslc == 362 *Should go for 64*
                        * Temporary files were not being combined, see below
                        * Included [. /home/igrigore/fsl/etc/fslconf/fsl.sh]

Concatenating temporary files: - Done by [possumX_postproc.sh]
                               - Was not doing so automatically before, had to chmod +x

Returning to fat: Try fractional approach: * Fat/(Fat + Water) => Feed as additional tissue segmentation
                                           * Try on full brain simulation

Full test TS: [~/possumdevpedro/bin/pulse -i ~/Inputs/ones_phantom_3D_MNI.nii.gz -o ./pulse --slcthk=0.001 --seq=ge --dx=0.001 --dy=0.001 --nx=256 --ny=256 --te=0.004 --tr=0.03 --gap=0.001 --numslc=362 --bw=840000 --angle=40]
              [~/possumdevpedro/bin/possumX /cluster/project0/GIF/Pedro/FullTest/ -n 100]

Full test MPM: **WIP**: - Getting NEWMAT::SubMatrixDimensionException error consistently
                        - Tried running outside of cluster to no avail either
                        - Try again with fewer nodes

Locally run possum + matrix:
[~/possumdevpedro/bin/possum_matrix -m ./motion -p ./pulse -o ./mainMat]
[~/possumdevpedro/bin/possum_working -i ./ones.nii.gz -w ./MPM.nii.gz -p ./pulse -f ./slcprof -m ./motion -o ./testSignal -e ./mainMat]





# Log: 27.06.18

Aims: - Check on output of Full SHR brain (FullTest folder)
        - Reconstruct brain using possumX_postproc
        - See in fsleyes
      - Make progress using fat files ~ [Fat_TS.nii.gz]
      - Successfully run full MPM sim

- Full SHR TS brain simulation: * Transverse plane seems to have been simulated great, expected resolution
                                * **However** Sagittal and axial planes seem squashed
                                  * Could be due to excessive slice thickness *MNI brain seems to have a thickness of 0.0005*
                                  * Or excessive slice gap *Maybe try zero intead of 0.001*

- Full SHR TS brain simulation, second attempt: * Re-ran pulse sequence with: [gap=0, slcthk=0.0005]
                                                <Errors>: * Consistently obtained Eqw == queued error waiting
                                                  * Probed with [qstat -j JOBID | grep error] and requed with -cj option
                                                  * However, seems like this did not work, did **NOT get expected temp. files**
                                                  * Cancelled task, re-ran with

- Full SHR MPM brain simulation: <Errors>: NEWMAT::SubMatrixDimensionException error obtained in all error log files
                                   * Probed by adding messages from start of MAIN LOOP to just after voxel calculation
                                   * Found that **MPM option was NOT being input**
                                   * Problem lied in possumX_MPM: Was re-writing command when checking for ONES instead of adding to it
                                                  <command = -i ones> instead of [command = $command -i ones]
                                   * Rectified problem: Now running appropriately





# Log: 28.06.18

# MRes Report plan [MEETING] ##PLAN ##REPORT
Requirements: ~15000 words total
              ~30 pages total *Actually ~70 pages lol*

Layout: - Introduction: * Brief background of MR -Shaihan Malik
                        * Description of problem in context
                        * Proposed solution
                        * Scope of project

        - Background: * MRI [Workings of MR, MR parameters, pulse sequences]
                      * MR Simulators [Static, Bloch equation, features and limitations] Fourier based solvers - beyond POSSUM
                      * Models of image synthesis (MachLearn)- Look into papers: Contrast mapping

                      *Up to here constitutes the literature review: ~12 pages*

        - Methodology: * Want to assess which simulation method is best
                       * Changed POSSUM to allow for MPM assignment
                       * Want to compare output of different methods
                       * Want to observe changes given features of different methods (e.g. POSSUM can simulate fat)
                       * Mention choice of sequences, justification
                       * Sequence optimisation: How and why
                       * Bulk assignment vs parametric maps: Process of creation

                       * Creation of TPM maps
                         * How simulator can use both: Curve matching
                         * Include "results", include **factual discussion**
                       * Fat integration, volume creation, validating fat shift, parameter sequences [Sanity check]
                       * Qual + quant of produced images: SNR/ CNR vs comparison to GT

        **NOTE** No need for results section
        - Results: * SNR + CNR sequence optimisation
                   * Large table of images comparing different simulation methods, sequences etc.
                   * Validation [SSIM (to GT), qualitative, etc.]

        - Discussion: * Assess which method is best [Computation time, similarity measures etc.]
        - Future work: One page
        - Conclusion: SE

        - Overfit model on data vs real

[NOTES]
- Re-started normal SHR TS POSSUM on cluster due to yesterday's errors: 100 processors
- Submitted SHR MPM POSSUM on cluster as per yesterday's notes

- Also altered props: **mpm_maps(xx, yy, zz)** -> **mpm_maps(xxx, yy, zz)**
  - Remains to be seen if this was correct, does not seem correct in retrospect
  - Want to check the properties of each voxel, not skip according to processor number
    - Validate by comparing against static equation *qualitatively might also suffice*





# 02.07.18

Aims: - Combine MPM cluster output together + analyse (qualitatively)
      - Decide if pulse + MPM choices were therefore chosen correctly: If yes then run for TEs + TRs
      - Formulate general MRes report plan

[NOTES]
- 46th node for MPMTest did not finish running:
  - "Moved" 100th output to 45th so possumX_postproc can run (needs sequential, no gaps)
  - Consider upping nodes to 150/ 200: Faster job completion should outweigh wait time
  - Need to move zeromotion to cluster folders *not sure what 'motion' is right now*
    - Double checked by using cat: Identical to zeromotion, so no need to change anything

- Sequence notes: https://radiology.ucsf.edu/sites/radiology.ucsf.edu/files/wysiwyg/research/nelson/Cha_07_Differentiation_of_glioblastoma.pdf
                  - TR = 34ms, TE = 8ms *paper*
                  - TR = 30ms, TE = 4ms *current use*
                  - Can try: * TR = 10, 20, 30, 40 ms
                             * TE = 2, 4, 6, 8 ms

- **NOTE** Need to investigate relation between slice thickness + number of slices
  - Actually, know that slcthk = 0.5mm is required (as per MNI brain)

- Pedro TS: * Affinely registered to MNI brain
            * Affinely registered PMs to these
              **WRONG** PMs clearly warped, maybe not correct to do this
                        Tried rigid instead, seems better but not 100%
            * Tried creating TS w/ fat
              * Looks off, fat overlap with intracranial region
              * Fat created by rigid, TS created by rigid *Maybe need to affinely create Fat*

[FILES]
Created multiple folders for the TE + TR simulations
TS_Pedro_TR# * TE2
             * TE4
             * TE6
             * TE8 *May not be viable*





# Log: 3.07.18

Aims: - Run remainder of MPM tasks, i.e. TRs + TEs
      - Run remainder of TS tasks, i.e. TRs + TEs
      - Properly add fat as tissue type to TS (+ MPM)
        - Run full brain sims for these

[NOTES]
- Started running MPM task for TR = 30ms, TE = 4ms (standard) *T1 weighted*

- Ideas for fat registration: 1. Register Fat segmentation to other segmentations *RIGIDLY*
                              2. Register (non-F) segmentations to MNI *AFFINELY* *Already done*
                              3. Compose transforms
                              4. Resample + add Fat to other segs

- Did exactly this **BUT** had to **register water instead**: * Not enough detail in fat image *Just outline of cranium*
                                                              * Registered Water to Segs
                                                                * Composed transform w/ Segs to MNI transform
                                                                  * Registered fat to segs using composed transform

- Create own package for nifty functions?

[Current tasks]
- MPM | TR 30 | TE 4 | NF
- TS  | TR 30 | TE 4 | NF
- TS  | TR 30 | TE 2 | NF





# Log: 04.07.18

Aims: - [D] Check on Pedro TS (TR = 30, TE = 4) in [TS_Pedro]
        - <N> Run remainder of TS sequences
      - [D] If Pedro MPM (TR = 30, TE = 4) runs to completion, check image
        - <N> Run remainder of MPM sequences
      - <N> Properly begin writing report
      - [D] Write bash script that sets up possumX environment automatically

[NOTES]
**MAKEFILE** Need to make changes in possumdevpedro NOT in /bin
  - Makefile copies files from possumdevpedro INTO /bin
  - Therefore if change bin directly, then overwritten by what's in possumdevpedro

**NOTE** Noticed that memory requests == 8G instead of 3.5G (What I set in fsl_sub)
  - possumX was running fsl_sub from {FSLDIR} instead of {POSSUMDIR}
  - Changed possumX to redefine {FSLDIR} == {POSSUMDIR}
  - Ran Pedro TS (TR = 30, TE = 4) again: * Checked memory by using qstat
                                          * Checked directory fsl_sub being called from

**NOTE** Disallowed certain nodes due to consistent poor performance: Cheech + Allen

- Pedro TS (TR = 30, TE = 4) did **NOT** run appropriately
  - Some diff_proc files were very incomplete
    *Could likely be due to fsl_sub error above, not enough time being allocated?*
  - Re-ran this task in proper folder TS_Pedro_TR30/TE4/
  - Began running task in TS_Pedro_TR30/TE6/ using NEW fsl_sub
  - Began running task in MPM_Pedro_TR30/TE2/ using new fsl_sub





# Log: 04.07.18

Aims: - TBD
      - TBD
      - TBD
      - TBD

# Log: 14.07.18 (Fuck)

Aims: - Generate "corrected" MPM
      - Run cluster for all TEs and TRs required: TS, TS_fat, MPM, MPM_fat
        - Check outputs of currently running MPMs
        - Remember that script was created for "slow" nodes

# Log: 16.07.18

[SUMMARY] **SCRIPTS** ##Scripts
- Created scripts for setting up simulation environment: * FatPossumXSetup.sh [SimType]
                                                           *For setting up fat environment*
                                                         * PossumXSetup.sh [SimType]
                                                           *For setting up non-fat environment*
                                                         * PulseMaker.sh: [SimType] [TR] [TE] [FullDir]
                                                           *Creates pulse files given TS/ MPM, TR and TE*
                                                         * SingleSubmission.sh [SimType] [ID] [ProcTot] [FullDir]
                                                           *Re-submission script for single task if slow node was used*
                                                         * fchecker.sh [SimType] [ProcTot] [FullDir]
                                                           *Checks logs for errors, re-submits those jobs*

- Ideal params: gap = 0

**ERROR** Problem w/ certain sequences: TR + TE choice not accepted by pulse.cc
          Seems to be due to <TG> param: == risetime * 100 *No idea why massive scaling, have emailed Mark*
          Prevents any sequences that use TR < 25 ms: Clearly does not reflect reality

[TO-DO]
- Create proper TS maps: Currently have isolated just GM, WM + CSF: Jorge's maps contain far more
- Create "corrected" MPMs w/out erroneous values
- Work on report

[NOTES]
- Crusher extremely long time explanation: * Long enough to guarantee spins can dephase ##Crusher
                                           * Typically fMRI does not use low TRs, therefore "good enough" for them
                                           * Adjust the scaling to, e.g. 30 *15 was chosen actually*

- TS problem: Volume 4 (Deep GM) + Volume 5 (Brain stem) added to normal GM/ WM segmentations

- Re-created TS (+TS_Fat) and transferred to cluster
  - TR30 dir was **NOT** changed yet due to sims currently residing there
  - All other dirs changed accordingly
  *Still need to check if MRpar in TS_Fat is correct version (Not 1.5T version)*

- Began running FB sims for TR20 + TR40, TE2, TE4, TE6
  - <P> n = 150
  - <P> BW = 300000
  - **Need** to check if allen nodes are slow (flag in fsl_sub was incorrect, cannot take multiple separate arguments)





# Log: 17.07.18
Aims: - Re-run failed nodes: Those with errors + allen nodes
        - Maybe automate process somehow
        - Find way to get output into appropriate folder (Instead of just putting into home)

# Jorge Meeting Notes
- To do: * Find out cause of "zipper artefacts" *Present in BOTH TS + MPM Possum*
         * Email Dave about bloom/ Bias field effects present in PD image *Take SS from FEyes*
           * Also ask about getting SPGR images for validation purposes
         * Try to get FAT images, too

[NOTES]
- Created <fchecker.sh>: * Checks simulation logs directory for errors, re-submits jobs if found
                           * Looks through those files w/ "e" in it (denoting error)
                           * Checks if file is empty (non-empty = error present)
                           * Checks digits after dot to find job ID
                           * Calls <SingleSubmission.sh> w/ job ID for each error found
                           *Maybe should also remove relevant log file in case errors pop up later*

- Need to look through Dave notes on fmri toolbox: **Choose 3D EPI, series 13, 22 images** (2nd data set)
  *Should be in old notepad, last page*

**BANDWIDTH CHOICE**: Was using extremely large (unfeasible) values  for BW *840000Hz, 300000Hz*
                      Created new directories to test out lower BWs (60000Hz)
                      Currently created [New_TS_Pedro_TR20] + [New_TS_Pedro_TR40] + repurposed [TS_Pedro_TR30] for this
                      Started running task for [TR=30, TE=4] for new BW *Can compare with existing output*

- Unsure if TRslc makes a difference (seems to be set automatically)
  - Created new directory, [LargeTRslcTestTE4] in [TS_Pedro_TR30] to test "expected" TRslc (i.e. TR * NumY)
  - Can compare to other two outputs (High BW + Normal BW)





# Log: 18.07.18 (post)

- Ran out of space in FULLBRAINSIMS dir
  - A lot of tasks could not write output to folders, ruined
  - Deleted most folder contents

- Re-ran TE4 + TE4_LargeTRslc
  - As well as some TR20 + TR40 + Fat
  - 100 processors to cut down on storage usage





# Log: 19.07.18
Aims: - Check output of yesterday's jobs
        - Mainly TR30, TE4s
      - Re-make PMs (maybe just PD if possible) using Dave's suggestion
        *B1 bias correction option + 3D EPI: Series 13, 22 images*
      - Work on SNR + CNR scripts
      - Report write-up (LR section)

- PD map high error (> 10%) as identified by hMRI toolbox *Suggests further error due to motion perhaps*
  - Motion seems to be present still
  **FIXED BY DT** Meeting tomorrow to go through process again

- Fat maps: TR30, TE4: * No success, map is identical to no fat map
                       * Checked if possum considering it (not skipping): <NOPE>
                       * Checked if possum_working was being used instead: <NOPE>
                       * Checked MRpar array fat values: [YES]
                         * Set to ms instead of seconds





# Log: 20.07.18
Aims: - Type up MPM creation process
        - Environment: [C:\Users\pedro\Desktop\Project\hMRI second try\Nifties]
      - Check on fat TS sim. [TR30, TE4]
      - Meeting: * Ask about what toolbox is doing
                 * Find out mistakes
                 * Ask about adaptive filters (filtering out noise in T2s map)

# Dave Thomas meeting
- Could not figure out what issue was: * Updated SPM
                                       * Re-ran DICOM -> Nifty
                                       * Checked jsons for 13
- Try running again w/ CURRENT setup
  - If no progress, re-download hMRI toolbox and try again
- Received DT MPMs to work with in the meantime

[NOTES]
- Re-created MPMs using DT PMs
  - Re-registered them
  - Can re-use for simulations

**T1W GRE**: Use 70 degrees for angle
**T2W GRE** Use 20 degrees for angle

- Tried re-running SPM w/ new version: Still same result (based on uncertainty error)
  - Should try qualitatively evaluating PDs
  - Try updating hMRI and running again

# Next week aims:
- Run ALL MPM + TS
  - T1w + T2w
*Check Fat output over weekend*
- Finish lit. review part of report





# Log: Weekend of 21.07.18
- Tested fat sims: * No discernible effect in TS sims *Maybe due to lack of skull in TS*
                   * Very noticeable effect for MPM: Tried low res sim(128 x 128 x 181), set SHR sim running
                     * All for T1W sequences, where effect should be more noticeable





# Log: 23.07.18
Aims: - Visualise MPM_fat output
        - If effects too pronounced, tone down CS
      - Lit review writing
      - Set other sequences running

[QMAPS DERIVATION NOTES] ##QMAPS
# Quantitative multi-parameter mapping of R1, PD , MT, and R2* at 3T: a multi-center validation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3677134/pdf/fnins-07-00095.pdf

- Signal depends on: [RC_Sen] * [PD] * [ANGLE] * [e^TR/T1] * [e^TE/T2s]
- Acquire three different ME FLASH scans: * T1-weighted: 18.7/20, 6 echoes (2.2 - 14.7)
                                          * PD-weighted: 23.7/6, 8 echoes (2.2 - 19.7)
                                          * MT-weighted: 23.7/6, 6 echoes (2.2 - 14.7) + off-res Gaussian RF
  - 1mm resolution
  - High BW: Minimise CS + Off-res
  - RF-transmit field: 3D EPI (SE + STE)
    - Geo. distortion + Off-res: Static B0 acquisition ~ 2D FLASH

[METHOD]
- SPM8 (12 in my case)
- Mean of T1w, PDw, MTw from 6 ME w/ smallest TE
  - Use these to calculate: * MT sat
                            * R1 *apparent*
                            * Signal amplitude *prop to PD: SANS RF inhomo, R2s effects*
                            * R2s *from ln of Signal @ echo times: Lin Reg*

[R1 Map]
- From R1_app: * Correct for RF inhomo
               * RF spoiling

[PD Map]
- From Sig. Amp.: * Adjust for global + receive sens.





# Log: 24.07.18
Aims: - Keep Lit. review writing
      - Run sims *Predicated on DT's response on sequence params.*

- Multiple errors (90 - std::bad_alloc) when submitting jobs
  - No idea what went wrong

- Changed possum_working.cc to lower effect of CS (by 10)
  - Running one job: [/cluster/.../MPM_Pedro_TR30_Fat/zeroPone/]
  - Previous image clearly FAR too affected by CS [Test_Sans_3]





# Log: 25.07.18
Aims: - Set multiple (~8 tops) running
      - Check on outputs (CS particularly)
      - Report *Assume all days include some amount of report writing from now on*

[JOBS]
- SET [MPM_Pedro_TR20] jobs running: * TE 4 ms
                                     * TE 6 ms
                                     * TE 8 ms
                                     * TE 10 ms

- CURRENTLY RUNNING: * [zeroPone]~ MPM TR30, TE4, FAT

[NOTES]
- Noted that some jobs will fail within a few minutes of running
  - Wrote script: @timercheck.sh (in ~/bin/): 1. Look at all log files *.o*
                                              2. Those that haven't changed in > 2hrs edit the corresponding error file *.e*
                                                2a. Add line to file to make size non-zero
                                              3. Can now submit new jobs by using @fchecker *Checks error files, submits if size > 0*





# Log: 26.07.18
Aims: - Check on outputs (CS particularly)
      - Streamline error checking in POSSUM sim. dirs.
      - **Aim to submit new set of tasks every day**: If staggered then should be no mem. problems.

[JOBS]
- SET [MPM_Pedro_TR10] jobs running: * TR 10, TE 4 ms
                                     * TR 10, TE 6 ms
                                     *Not sensible to run any higher*

      [MPM_Pedro_TR40] jobs running: * TR 40, TE 4 ms
                                     * TR 40, TE 6 ms
                                     * TR 40, TE 8 ms
                                     * TR 40, TE 10 ms

      [T2sW] jobs running: * TR 300, TE 25 ms
                           * TR 450, TE 25 ms
                           * TR 600, TE 25 ms
                           * TR 750, TE 25 ms
                           @cluster/project0/possumBrainSims

- CURRENTLY RUNNING: * [zeroPone]~ MPM, TR30, TE4, FAT
                     * [MPM_Pedro_TR20]~ ALL TEs (4, 6, 8, 10 ms)

[NOTES]
- Wanted script to check what tasks have finished running
  - Wrote script: @diffchecker.sh (in ~/bin/): 1. Look at all diff_proc files *signal_proc_*
                                               2. Take last digit(s), compare against full sequence [1 : 1 : Nproc]
                                               3. Print out missing files (i.e. uncompleted tasks)
- Started writing Image Synthesis section: * Registration based approaches
                                           * Intensity based approaches
                                             * Mach. Learning algorithms

- Set T2s jobs running for constant TE (25 ms), variables TR (300, 450, 600, 750 ms)
  - Should provide useful preliminary contrasts to look at
  *TE 25 ms is around the clinical scan value*

**NEWDIR** @cluster/project0/possumBrainSims ##T2WDIR
- Folder structure: * T2W
                      * MPM
                        * Pedro_TR300
                          * TE15
                          * TE25
                          * TE35
                          * TE45
                        * Pedro_TR450
                          * <CP>
                        * Pedro_TR600
                          * <CP>
                        * Pedro_TR750
                          * <CP>
                      * TS
                        * Pedro_TR300
                          * <CP>
                        * Pedro_TR450
                          * <CP>
                        * Pedro_TR600
                          * <CP>
                        * Pedro_TR750
                          * <CP>





# Log: 28.07.18 (pre-emptive)
Aims: - Re-run validation simulations for full brain
        *Re-read notes for these: #EXPO_INIT*
      - Analyse outputs of TR40, TR20
      **RECONSTRUCT THESE ASAP TO NOT RUN INTO SPACE ISSUES**

[NOTES]
- Reconstructed 0.1 MPM Fat sim.
  - Looks even worse than before
  - Massive artefacts
  - Should debug: Print out CS effect (e.g. frequency shift)

- Reconstructed MPM TR20 sims.
  - Seem good, but "zipper" artefact present
  - @Email: Emailed Mark about this
  - Set NEW sequences running in [MPM_Pedro_TR30] * TE6 is running at 40kHz
                                                  * TE8 is running at 40kHz + slcThk = 1mm (Instead of 0.5mm)
  - 40kHz on lower end of bandwidth, should determine if issue is BW related or not

- Queues taking a long time
  - **Cancelled** [MPM TR750 TE25] for now to get post processing done on others

**Problem with postProc submission**
- Memory issues running on login node
- Created a script @submitPP_creator.sh (~/bin/)
  - To submit with qsub: [submitPP_creator.sh Directory nProc]
    - Automatically creates postproc script and qsubs it (12G mem)

[JOBS]
- SET [MPM_Pedro_TR30] jobs running: * TR 30, TE 6 ms *BW test*
                                     * TR 30, TE 8 ms *BW + slcThk test*

- CURRENTLY RUNNING: * [MPM_Pedro_TR40]~ MPM, ALL TEs (4, 6, 8, 10 ms)
                     * [MPM_Pedro_TR10]~ MPM, TE 4, 6 ms
                     * [T2W/MPM_Pedro/.../TE25]~ MPM, ALL TRs (300, 450, 600, 750 ms)





# Log: 30.07.18 (post)
- Mostly did report writing
- Set a few different things running: * Most of the MPM T2W TE25s have finished
                                      * Started running some, corresponding, TS
                                      * Need to run MPMs of same TE as well for T1W





# Log: 31.07.18
**CRUSHER TEST** Can see "zipper" artefact in images
  - Running a test in [/cluster/project0/GIF/Pedro/MPM_Pedro_TR40/TE4_crusher] w/ original crusher time
  - Can compare directly to MPM TR 40, TE4 image (That has the zipper artefact)

- Don't forget to create proper validation!
  - Create simple + non-simple (artificial) MPMs
  - No registration should be necessary: Already in MNI brain space
  - Run for two T1W sequences + two T2W sequences
  *Need to determine what "t" to choose for MPM creation*

- Ended up making BOTH TE and TR artificial MPMs
- Found in $FULLBRAINSIMS/Artificial_MNI_MPMs
  - Added everything into each folder except for PULSE





# Log: 01.08.18 ##ERORES ##ERO_RES
Aims: - Finish MPMa stuff
      **Check CRUSHER**
# Meeting
- Zipper images are fucked
  - T2s PM is far too noisy
  - Ask about re-aquisition w/ DT

- Perform SNR + CNR: * Basal Ganglia
                     * Cortex
                     * WM + GM *Zipper images*
  - Use erode function to "erode" segmentations: Remove PV effects for "true" SNR of tissue
    - Part of [FSLMATHS]

fslorient -setsformcode 2 image_abs.nii.gz *Turns on flag that lets us change origin of image*
fslorient -setsform -1 0 0 126.5 0 1 0 -144.5 0 0 0.5 -71 0 0 0 1 image_abs.nii.gz *Align "zero" of image* *Use on POSSUM outputs*
fslorient -setsform -1 0 0 127 0 1 0 -145 0 0 0.5 -71 0 0 0 1 image_abs.nii.gz **BETTER version**





# Log: 02.08.18
Aims: - **Main** Fix T2s map: * Do not use abs.
                              * Can just resample previous transform, will use full process however

      - Write + test SNR + CNR code on Jog images
        - Investigate variances
        - Plot histograms
        - SNRs first, CNR == pairings between regions

[NOTES]
- WM, GM, CSF: In Jorge's segmentations (conc)

- Basal Ganglia composition: * Caudate nucleus <37> <38>
                             * Putamen <58> <59>
                             * Ventral striatum: + Nucleus accumbens <24> <31>
                                                 + Olfactory tubercle [??]
                             * Globus pallidus [??] <56> <57>
                             * Ventral pallidum [??] <61> <62>
                             * Substantia nigra *Ventral DC* <61> <62>
                             * Subthalamic nucleus *Ventral DC* <61> <62>

- Cortex: * Calcarine cortex <109> <110>
          * MFC medial cortex <141> <142>
          * SMC supplementary motor cortex <193> <194>

 + Gyrus: * <101> <102>
          * <105> <106>
          * <107> <108>
          * <123> <124>
          * <125> <126>
          * <129> <130>
          * <133> <134>
          * <135> <136>
          * <137> <138>
          * <139> <140>
          * <143> <144>
          * <145> <146>
          * <147> <148>
          * <149> <150>
          * <151> <152>
          * <153> <154>
          * <155> <156>
          * <161> <162>
          * <163> <164>
          * <165> <166>
          * <167> <168>
          * <171> <172>
          * <177> <178>
          * <179> <180>
          * <183> <184>
          * <191> <192>
          * <195> <196>
          * <197> <198>
          * <201> <202>
          * <205> <206>
          * <207> <208>





# Log: 03.08
Aims: - Correct T2* map
      - Create (eroded) regions from segmentations: * Deep grey matter
                                                    * Cortex
                                                    * WM, GM, CSF
      - Check on jobs: [MPM_TR40_TE4_PDdel_fp]
                       [MPM_TR40_TE4_PDdel]
                       [MPM_TR30_TE10]

[NOTES] ##T2sCorrection ##RECTIFIER
- T2* map correction
  - Tried setting negative values to log. range btwn 1e-5 & 1e2
  - Ran Jog sim. for T2sW seq (TR=600ms, TE=25ms)
  - Best sims for 1e0, 1e1 *Find middle ground*
  - Transition too abrupt, use rectifier function to "smooth" `log(1 + e^x) + c`

- Region erosion
  - Use binaries: Threshold at 0.5

- SNR + CNR
  - Parametric -> non-parametric stats
  - i.e. consider using <median / IQrange> for SNR
    - Best for when dist. isn't completely Gaussian *PV effects*

**NO LEAVING UNTIL COMPLETED**
- Ended up using "replacing" value = 4 (i.e. negs -> 4 s^-1)
  - Used `log(1 + 10e^(0.3x))`: * Needed transition to be smooth enough across range of negative values *i.e. ~ -20 to 0*
                                * Factor: Changes y-intercept
                                * Expo: Changes rate of slope





# Log: 4.08.18
Aims: -

[NOTES]
- Reconstructing: * Validation T1W/TE12
                  * TE4_PDdel
                  * PRELIM TE4_PDdel_fp

- New MPMs w/ R2* correction: * Much better than previously
                              * Re-simulated some Jog images
                                * T1W almost identical (Have artefact on top of head)
                                * T2W **FAR** less noisy
                                * See [C:\Users\pedro\Desktop\Project\Jog Tests Old vs New MPM]





# Log: 5.08.18

[NOTES]
- Set most validation tasks running
- Moved **UN-CORR** version of new (T2*) adjusted MPM to [T2W] sim dir. + ClusterFiles/MPM
  - Started re-running all TE25 T2W tasks *might take a while to queue them up*
- Currently reconstructing TS TR600 TE25

- TS T1W missing: * TR30: TE6
                          TE8
                          TE10
                  * TR40: TE6
                          TE8
                          TE10
                  * TR20: TE6 (Missing signal_proc_0)
                          TE8 (Missing some files)
                          TE10 (Missing some files)

- Started writing SNR + CNR script: [SNR_CNR.py]
  - Relevant regions: DGM, Cortex, GM, WM, CSF





# Log: 6.08.18
Aims: - Finish SNR + CNR code
        - Test on Jog images *Remember to check TS Jog*
        - Test on MPM + TS POSSUM available images

[NOTES]
- SNR + CNR plotting notes: * Plot all TEs per region? (hist)
                            * Plot average for each TR? (hist)
                            * Should work on SNR plot





# Log: 7.08.18
Aims: - Fill in T1W gaps (TR30)
      - Submit missing TS jobs
      - Refine SNR + CNR code

[NOTES]
- Set following jobs running: * MPM TR30: TE8
                                          TE10 (Missing one signal_proc)
                              * TS TR20: TE6 (Missing signal_procs)
                                         TE8 (Missing signal_procs)
                                         TE10 (Missing signal_procs)

- Should test if noise is actually getting added: Use signal2noise and compare

# Meeting notes:
- Look at k-space
- Use new CSF to compare
  - Plot histograms to compare
- Investigate further maps





# Log: 08.08.18
Aims: **VALIDATION**





# Final compiled notes
- Validation status: * NS MPM T1W: Real + imag @comic
                     * NS MPM T2W: Real + imag @comic
                     * S  MPM T1W: Real + imag @comic
                     * S  MPM T2W: Real + imag @comic

                     * NS TS T1W: Real + imag @comic
                     * NS TS T2W: Real + imag @comic
                     * S  TS T1W: Real + imag @comic
                     * S  TS T2W: Real + imag @comic Potentially re-do with JOG

- T2W status: * MPM TR300 TE25: Error: *Re-running 37 + 40*
              * MPM TR450 TE25: Real + imag @comic
              * MPM TR600 TE25: Real + imag @comic
              * MPM TR750 TE25: Real + imag @comic

              * TS TR300 TE25: Real + imag @comic
              * TS TR450 TE25: Real + imag @comic
              * TS TR600 TE25: Real + imag @comic
              * TS TR750 TE25: Real + imag @comic

- SNR + CNR discussions: Median vs mean (quantitative stats vs not) {Report}
- Correct bias field in real images [DONE] *mri_nu_correct.mni --i TR20_TE4.nii --o TR20_TE4_BC.nii*
- Register TS to real images (intead of current) [DONE]
- Add Rician noise to all plots [DONE] ##RICIAN
- Consider re-registering TS to Real images (Use iso option, internal)
- Add noise to TS plots, too
- Re-formulate rician noise: Signal dependant
- Larger future work section: * What to be done in terms of simulations (i.e. source from multiples, etc)
                              * What big picture is: Image synthesis, very next step (learning between two TR/ TEs)

- Discussion: Directly compare POSSUM and Static equations <N>
- Get Segmentation approach from Jorge (NeuroMorph) [Done: Geometric geodesics paper] ##SEGMENTATION_REFERENCE
- Niftireg citation [Kinda- Linked the sourceforge]
- Maybe add minisection about contrasts [Added figure, could add more detailed discussion/ plots]
- Add "big picture" image to start of methodology <N>
- Appendices: * Non-included pictures
              * MPM creation code (maybe)










# Log: 25.09.18
Aims: - Read simulator papers: * See what's out there
                                * Compare features
                                * Investigate useability/ open-sourceness
      - NiftyPipe tutorials: * All online

[NOTES- Simulators]

Fast Realistic MRI Simulations Based on Generalized Multi-Pool Exchange Tissue Model (MRiLab): https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7676360
- Generalised tissue model: Multiple exchanging water + macromolecular proton pools *Instead of* system of independant isochromats (Like most other sims.)
- Employs GPU parallelization: ~ 200x faster than CPU
- Cross-platform, open-source, extensible

[Introduction] * Good for rapid prototyping, evaluation of techniques
               * Initially: Proton isochromats -> Analytical signal expressions
                 * Now: Numerical modelling based approaches
                 * Limitations (others): Use of simplified tissue representations
                   * Single compartment model for protons, instead of more realistic multiple, interaction, model
                   * MRI signal cannot be described fully with SCM
               * Multi-pool models: Useful for characterising tissue composition, microenvironment, microstructure (not only path.)
                 * e.g.: qMTI, M-C relaxometry, CEST
                 * Multiple images -> qMAPs of tissue compartments (vs single compartment T1/ T2 MAPs)
                 * e.g. fatty liver inflitration w/ Fat + Water Model.
               * Macroscopic effects: Modelling of PV of CSF
               * Computation: Burden for 3D + multi-pool
                 * Have to resort to clusters, typically
                 * Not needed with MRiLab

[Theory] * Have to account for multiple proton sources; bound + unbound; + PV effects
         * Consider Nf free protons + Nb bound protons, connected by magnetisation pathways
           * Choose [config. of model] + [params.] to represent each tissue type
            *Pool number, type, pathways* + *relative fractions, T1/ T2, CS*
        * Employ (finite differential) **Bloch-McConnell** equations in RF for Nf protons, **MT formalism** for Nb protons

[Objects] * Anatomy represented as collection of voxel elements
          * For each voxel, have tissue-specific config. + model params.
          * PV: Discretize object at finer level than image resolution

[Experiment] * Maps of B0: Transmit + receive
             * Imaging gradients
             * Build PS graphically
               * Can account for external effects, e.g. spoiling via augmentation
             * Once all set up: Solves MP ODEs
               * Discrete time-solutions: Rotation + exp. scaling matrices @ each time point through the PS
               * Elements in object independant: Use parallelisation of GPUs, using [CUDA]

[Methods + Results] * Matlab implementation
                    * C++ implementation of kernels

[Discussion] * Distinctions: Multi-pool + computational engine

[Conclusion] * Flexible representation of tissues
             * Multi-pool approach
             * Flexible, open-source

- Downloaded MRiLab: [Project/MRiLab]
  - Execute from MatLab directly by calling <MRiLab>


# Poster pitch
Project summary:
* MRI is largely a qualitative imaging technique, which is to say if you perform a scan in different scanners there will almost certainly be differences between the obtained images. The ultimate goal of the project is to be able to learn to transform between different contrasts and different scanners. This can make studies that require precise volumetric measurements difficult because it isn't straight forward to account for these differences. By employing simulators we can hopefully train networks to learn between modalities, and therefore correct for most of the observed differences.

* So far I've worked more on the simulation side of things, testing the performance of different simulation approaches and seeing how they compare against real images

CDT Courses:
* Computational MRI: I should say that I have no background in Medical Imaging, so I thought that I gained a lot from the course, the best way to understand a topic is by coding it, though word of warning it was a lot of work, Gary will work you to the bone

* The Research Software Engineering with Python I thought was a particularly good use of my time, don't be discouraged if you're not familiar with the language the focus of the course is more about how to write good code, how to use version control, how to properly test it more than anything else. IPMI I found extremely useful since I had barely heard of terms such as segmentation and registration

Overall advice:
* Don't get bogged down by coursework, try to steadily work on your project so you don't have to put all the work in at the end
* Goes without saying, try to set up regular meetings, don't be afraid to put forward new ideas or disagree
* Enjoy your cohort, you'll be spending a lot of time together so you might as well try to enjoy that time



# Log: 04.10.18
Aims: NiftiNet

[Intro]
Initial model experiments: * net_download dense_vnet_abdominal_ct_model_zoo *Abdominal CT 3D nifty*
                           * net_segment inference -c ~/niftynet/extensions/dense_vnet_abdominal_ct/config.ini
                             * Segmentation found: ~/niftynet/models/dense_vnet_abdominal_ct/segmentation_output/

[Overview] https://niftynet.readthedocs.io/en/dev/config_spec.html
[Model Zoo] https://github.com/NifTK/NiftyNetModelZoo/blob/master/README.md
- Workflow can be fully specified by NN application + config. file
                                  **net_run [train | inference | evaluation] -c <path/config.ini> -a application**
  - <Netrun>: Entry point for NN
    - train: Update underlying network model using DATA
    - inference: loading existing network model -> generating responses according to data
  - <Application>: Specify application (user.path.python.module.MyApplication) OR as ALIAS
    - Examples: * [Segmentation] niftynet.application.segmentation_application.SegmentationApplication OR net_segment
                * [Regression] niftynet.application.regression_application.RegressionApplication OR net_regress
                * [Autoencoder] niftynet.application.autoencoder_application.AutoencoderApplication OR net_autoencoder
                * [Generative adversarial network] niftynet.application.gan_application.GANApplication OR net_gan
  - <Overriding>: Instead of creating sep. file, can pass arguments to net_run to override certain params
    * --<name> <value> or --<name>=<value> to override <name> with <value>

[Configuration: Sections]
- Adopts .ini format, parsed by configparser, contains multiple sections
- Separated into [SYSTEM] and [NETWORK] sections
- If <train> specified, then [TRAINING] section is required
- If <inference> specified, then [INFERENCE] section required
  *Also need app. specific section for each app.*
  [GAN], [SEGMENTATION], [REGRESSION], [AUTOENCODER]
- Other section names: Treated as input data specifications *e.g. csv_file=FILE.csv*
  - These include: * csv_file: Input images
                   * path_to_search: SE
                   * filename_contains: Match keywords for FN
                   * filename_not_contains: Don't match keywords for FN
                   * filename_removefromid: Extract subject id, subt. from FNs
                   * interp_order: SE
                   * pixdim: Resample into specified voxel sizes
                   * axcodes: Axes reorientation
                   * spatial_window_size: 3 integers, input window size *e.g.: 64, 64, 1 for 2D slice window*
                   * loader: Specify loader, default tries all *e.g.: simpleitk*
  - Example
    path_to_search = ./example_volumes/image_folder
    filename_contains = T1, subject
    filename_not_contains = T1c, T2
    spatial_window_size = 128, 128, 1
    pixdim = 1.0, 1.0, 1.0
    axcodes = A, R, S
    interp_order = 3

[SYSTEM]
<cuda_devices>: Device visibility
<num_threads>: Number of preprocessing threads
<num_gpus>: Number of training GPUs
<model_dir>: "Working" dir.
<dataset_split_file>: File assigning subjects to training/validation/inference
<event_handler>: SE

[NETWORK]
<name>: Class from niftynet/network *niftynet.network.toynet.ToyNet*
<activation_function>: Type of activation *See SUPPORTED_OP*
<batch_size>: Number of image windows for processing at each iter. *inps = B_S x NGs*
<smaller_final_batch_mode>: Division errors *drop, pad, dynamic*
<decay>: Strength of regularisation *lambda*
<reg_type>: Type of reg. *L1 or L2*
<volume_padding_size>: SE *=M,N,L or =M if M=N=L*
<volume_padding_mode>: Type of padding, see:  https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html
<window_sampling>: Type of sampler *Uniform, weighted, balanced, resize*
<queue_length>: Window buffer size when sampling windows from volumes
<keep_prob>: Prob. that each element is kept if dropout enabled *Default 0.5, same as inference stage* If deterministic: = 1, If stochastic: = 0

[VOLUME-NORMALISATION] Intensity-based
(1) normalisation=True: Histogram-based norm.
https://niftynet.readthedocs.io/en/dev/niftynet.utilities.histogram_standardisation.html
*histogram_ref_file, norm_type, cutoff, normalise_foreground_only, foreground_type, multimod_foreground_type*
(2) whitening=True: Volume level norm. (I - mean(I))/std(I)
*normalise_foreground_only, foreground_type, multimod_foreground_type*

<normalisation>: Bool
<whitening>: Bool
<histogram_ref_file>: Normalisation param *use prior or create*
<norm_type>: Hist. landmarks (perc. or quart.)
<cutoff>: Inferior + superior hist. cutoffs
<normalise_foreground_only>: SE
<foreground_type>: Foreground only norm. *otsu_*
<multimod_foeground_type>: Combine foreground masks of multiple modalities

[TRAINING]
<optimiser>: SE, *See SUPPORTED_OP*
<sample_per_volume>: Samples from each volume
<lr>: Learning rate
<loss_type>: Type of loss function *Many types, depends on task*
https://niftynet.readthedocs.io/en/dev/niftynet.layer.loss_<TASK>.html
<starting_iter>: =0 (random), =-1 (latest checkpoint)
<save_every_n>: Saving freq. of training model *Final always saved*
<tensorboard_every_n>: Freq. writing to tensorboard + graph elements
<max_iter>: SE
<max_checkpoints>: SE

[Validation during training]
- Can have validation loops during training (Set validation_every_n)
  - Image list will be partitioned into training/ validation/ inference *according to exclude_fraction-for_validation and exclude_fraction_for_inference*
  - Generates CSV, mapping randomly to each of stages
  - Can be edited manually

[Data augmentation during training]
<rotation_angle>: Array, random rotation operation on volumes *Careful with dim.*
<scaling_percentage>: Array, random spatial scaling *(-n, n) == 0.n% to 1.n%*
<random_flipping_axes>: Axes flipping *0,1*

[INFERENCE]
<spatial_window_size>: Size of input array *e.g.: 64, 64, 64 for 3D vol.*
<border>: Crop border size
<inference_iter>: Specify trained model *=-1 for latest model in model_dir*
<save_seg_ir>: Relative to model_dir
<output_postfix>: Postfix on output FNs
<output_interp_order>: SE
<dataset_to_infer>: Which dataset to infer for *all, training, validation, inference*

[EVALUATION]
Evaluation of output against some GT
                    **net_run evaluation -c <PATH/CONFIG> -a <APP>**
Multimodal: net_run evaluation -a niftynet.apps.seg_apps.SegApp -c config/MM.ini
Reqs: * GT to compare against
      * Files to evaluate
<save_csv_dir>: SE
<evaluations>: List of metrics *dice, jaccard, n_pos_ref, n_pos_seg*
<evaluation_units>: How to perform (in case of segs) *foreground, label, cc*
https://niftynet.readthedocs.io/en/dev/niftynet.evaluation.segmentation_evaluations.html

[FILENAME MATCHING]
Should strive to use same unique subject identifiers across files:
- T1_001_img.nii.gz *Image files*
- 001_img_seg.nii.gz *GT*
- subject_001,training *CV*

Can do this automatically, too:
- Use config params: <filename_contains>, <path_to_search>, <filename_not_contains>
- Automatically appends new line to auto. generated .csv files
  - i.e. whatever is leftover after removing FN_con + FN_not_con

Subject ID extraction
- Can further remove strings by using <filename_removefromid> = STR


# Meeting notes: 5.10.18
given multiple inputs: Can segment? Pseudo ground truth from MPMs
Nvidia grant for GPU
Email Dave Thomas: MPMs from the Nick Weisskov (studies)
                   DRC clinical study
                   Any MPMs really


# TENSORFLOW: KERAS INTRO
import tensorflow as tf
from tensorflow import keras *High level API for TF*

[DATASETS]
Loading from keras example: keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
*Train set == 2 arrays*, *Test set == 2 arrays*

- Fashion dataset, 60000 training, 10000 test
- Images are 28 x 28
- Labels 0 - 9: Represent different items of clothing
- Class labels, accordingly: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

[PRE-PROCESSING]
- E.g.: Scaling pixel values to lie btwn 0 - 1 (vs 0 - 255)
  - **Training and Test should be pre-processed in same way**

[BUILDING THE MODEL]
- Build using keras
- Based on layers: Each layer extracts representations of data fed to it
- Sequential process: * Reformat data (e.g.: Flatten)
                      * First NN layer: 128 nodes
                      * Second NN layer: 10 node SoftMax, prob of data belonging to each label (sum to one)

[COMPILING THE MODEL]
Need to choose further settings:
- Loss function: Function wanting to be minimised *e.g.: L1/ SSD*
- Optimiser: How model is updated based on data + LF *e.g.: GD*
- Metrics: Monitoring training/ testing *e.g. accuracy*

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=tf.train.AdamOptimizer(),
#               metrics=['accuracy'])

[TRAINING THE MODEL]
- Sequential process: * Feed training data to model *train_images, train_labels*
                      * Have model learn associations btwn images & labels
                      * Test model predictions on test set *test_images*
                        * Compare against GT label set *test_labels*

- model.fit(train_images, train_labels, epochs=5) *Epoch: How many times going through dataset*
  - Loss and accuracy displayed as model trains

[EVALUATE ACCURACY]
- Compare how model performs on the test dataset
  - test_loss, test_acc = model.evaluate(test_images, test_labels)
  - print('Test accuracy:', test_acc)

# Overfitting: train_acc > test_acc
# Underfitting: train_acc < test_acc

[MODEL PREDICTIONS]
- Once model trained, can be used to make predictions
  - predictions = model.predict(test_images)
    - Outputs len N array, where N is number of labels, each entry == prob/ confidence
    - Use argmax to find most probable label: np.argmax(predictions[n])

- Single image prediction: * Keras models processed as batch
                           * Expand image dimensions: (28, 28) => (1, 28, 28) *np.expand_dims(img,0)*
                             * single_prediction = model.predict(img)

# TENSORFLOW: FURTHER
**NOTE**: Breaks when using version 1.10+ [pip install --upgrade tensorflow==1.10]
[EAGER EXECUTION]
- Enable more interactive frontend to TF
  - **_tf.enable_eager_execution()_**

[TENSORS]
- Multi-dimensional array: Data type + shape
- Equivalent tensor-specific operations: *tf.add(), tf.mul(), tf.square()*
  - e.g. tf.add(1, 2)

[NUMPY COMPATIBILITY]
- Differences vs np arrays: * Can be GPU backed
                            * Immutable (Don't change over time)
- Tensorflow operations auto. convert np arrays to tensors
- Numpy operations auto. convert tensors to np arrays
  - Can convert Tensors to np arrays by invoking .numpy() method

[GPU ACCELERATION]
- TensorFlow operations can be accelerated using GPUs
  - tf.test.is_gpu_available()
  - TENSOR.device.endswith('GPU:N') *Check if TENSOR is on GPU N*
- Explicit assignment: _with tf.device("CPU/GPU:0"):_

[DATASETS]
- Employs [tf.data.Dataset] API for dataset construction
  - If using EE, do not need to explicitly create an Iterator: [tf.data.Iterator]
  - Can create using slices/ full tensors using [Dataset.from_tensors] or [Dataset.from_tensor_slices]

  # EXAMPLE
  ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

  # Create a CSV file
  import tempfile
  _, filename = tempfile.mkstemp()

  with open(filename, 'w') as f:
    f.write("""Line 1
  Line 2
  Line 3
    """)

  ds_file = tf.data.TextLineDataset(filename)

[TRANSFORMATIONS]
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2) *Can apply multiple methods*
ds_file = ds_file.batch(2)

[SEQUENTIAL STUFF] **IF EE ON**
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

[AUTOMATIC DIFFERENTIATION] == Numerical differentiation
- Technique for optimising models
- APIs available for computing derivatives [.gradients_function(FUNCTION)] *similar to autograd*
  - Can define derivative function accordingly, e.g.:
  # return lambda x: tfe.gradients_function(f)(x)(0)

[GRADIENT TAPES]
- Reverse accumulation: Start from "outside"-most, and move inwards with partial derivatives
  - i.e.: dy/dx = dy/dw2 * dw2/dw1 * dw1/dx: https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
  - Start with derivatives wrt last "variable"
- If care about intermediate results, use tf.GradientTape
  # with tf.GradientTape(persistent=True) as t:
  #   t.watch(x)
  #   y = tf.reduce_sum(x)
  #   z = tf.multiply(y, y)
    - Guarantees that x is monitored and stored for later:
      - e.g.: dz_dx

[HIGHER ORDER DIFFERENTIATION]
- tf.GradientTape records gradient computations as well:
  - i.e.: Nested with statements:

  with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
      t2.watch(x)
      y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)
  d2y_dx2 = t.gradient(dy_dx, x)

[TRAINING BASICS: FIRST PRINCIPLES (NO KERAS)]
- Stateful vs stateless: Storing previous states (Python) vs NOT (TF Tensors)
- TF Variables are stateful however: [tf.contrib.eager.Variable(X)]

[LINEAR MODELS]
1. Define model
2. Define LF
3. Training data (TD)
4. Run through TD, use optimiser, adjust variables to fit data

- f(x) = x * W + b
  - Variables <W> & <b>
    - <W> = 3.0
    - <b> = 2.0

**MODEL**
- Define class: <VARIABLES> + <COMPUTATION>
  - Randomly initialise <VARIABLES>, since it is an initial model

**LOSS FUNCTION**
- How well does model fit?
- Define as function
  - Can be as simple as SSD

**TRAINING DATA**
- Synthesise TD + noise
  - E.g.: Input = tf.random(shape=[N])
          Noise = tf.random(shape=[N])
          [Output] = <Input> * W + b + <Noise>


**PROCESS**
- Visualise model: * Plot Training Data
                   * Plot Model Data
                   * Plot Loss

- Define training loop: Want loss to decrease, use gradient descent
  - [tf.train.Optimizer] Contains many such GD implementations
  - Do it from first principles as a first attempt
  def train(model, inputs, outputs, learning_rate):
  #  with tf.GradientTape() as t:
  #      current_loss = loss(model(inputs), outputs)
  #  dW, db = t.gradient(current_loss, [model.W, model.b])
  #  model.W.assign_sub(learning_rate * dW) *.assign_ == do operation and mutate variable*
  #  model.b.assign_sub(learning_rate * db) *.assign_ == do operation and mutate variable*

[LAYERS]
- Typically use tf.keras as HL API for NN
  - Benefit of custom layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers

[CUSTOM LAYERS]
- Seems complicated, revisit later

**NOTE**
Start Spyder in dev mode to allow for proper use of S-terminal
  - [set SPYDER_DEV=True]

[DATASETS] ~ tf.data.{Dataset}
- Module that allows for ease of data loading, manipulation, model insertion (e.g.: From CSV files)
- https://www.tensorflow.org/guide/datasets_for_estimators for further information


# Meeting points to bring up: 12.10.18
- Re-iterate "discriminatory" model we want to eventually develop
- Niftynet/ Keras/ TensorFlow
- TF Custom Layers: Relevance
- Funding for Coursera courses: Possible? Who to talk to
- Talked w/ Dave, going to send all types of data
- No response regarding computer parts

[TO DO] ##TODO ##Coursera
- Deep learning book: Chpts 5 (recap) - 9 file:///C:/Users/pedro/Downloads/deeplearningbook.pdf
- Stanford Deep Learning: http://cs230.stanford.edu/syllabus.html (Sans last section)
- Stanford CNN: http://cs231n.stanford.edu/ (After SDL)
  - TensorFlow tutorials (continue)
- NiftyNet tutorials
  - Specifically U-Net: http://www.miccai.org/edu/finalists/U-Net_Demo.html

[CUSTOM WALKTHROUGH: SIMPLE MODEL]
- Always DOs: * import tensorflow as tf
              * TFE = tf.contrib.eager
              * tf.enable_eager_execution
- Dataset download: * [URL] = http://www.etc
                      * concatenate: [URL_CONC] = tf.keras.utils.get_file(fname=os.path.basename(url),
                                                                          origin=train_dataset_url)

- Use pandas to read/ process CSV data: * import pandas as pd *pd.read_csv*
                                        * print(pd.read_csv([URL_CONC], nrows=5))

- Should explicitly label [column names] == **FEATURES**
  - e.g.: [CN] = ['length', 'width', 'density']
- Also label [classes] == **LABELS**
  - e.g. [ClN] = ['Bear', 'Cat', 'Pony']

- Create **DATASET** in correct format: **DATASET** = tf.contrib.make_csv_dataset(
                                                                                  [URL_CONC], *Dataset*
                                                                                  [BATCH_SIZE], *Number of TRAINING examples in dataset*
                                                                                  [COLUMN_NAMES], *i.e. features*
                                                                                  [LABEL_NAME], *i.e. classifications*
                                                                                  [NUM_EPOCHS]) *How many time to pass dataset to NN*

- Iterate through **FEATURES** + **LABELS**: FEATURES, LABEL = next(iter(**DATASET**))




**NOTE** [SUPER METHOD]
- Allows for skipping of referral to base class in initialisation
- Also allows for calling of init. of other inheritee classes, e.g.:

class T(Object):
  def __init__(self):
    print('This string originates from class T')

class A(T):
  def __init__(self):
    super().__init__()

a = A()
>> This string originates from class T
## Linkedin
- Settings -> Privacy -> Job setting preferences -> Let recruiters know etc.















# Meeting Notes: 23.10.18
- Jorge will get in touch with Frederik/ Rebecca to iron out supervisor details
- Another Alex email sent to chase her up on funding for PC
- Move away from fully connected networks (80s/ 90s concept: Universal approximator): Find most efficient way to approximate: Cannot possibly have a fully connected network for large image

- ImageNet paper 2012
- interpretable deep learning

- Re-iterate network we want to implement








# Meeting Notes: 30.10.18
GAN: Generative Adversarial Network
  discriminator -> Loss function
  Waterstein GAN
  F1 GAN
  Pitfalls of GANs: MICCAI paper: Distribution Matching Losses can hallucinate features in medical image translation
  CycleGAN
  Talk to (for implementation): Marta, Danielle, Kerstin






# Concise summary
Use network to learn imaging parameters from input images => Feed parameters into simulator to synthesise images of different contrasts
Potentially adapt this to GANs (Generative adversarial networks): Feed an image (e.g.: T1)
  => Generate new image (e.g.: T2)
  => Have a real T2 image
  => Train discriminator: Tries to establish which of the input images is the "real" image
  => If can distinguish then need to work on improving mapping (Useful for "dual gaussians")

"Ψ-CLONE has certain limitations which we would like to address in the future. First, it
requires a segmentation of the input image(s) to estimate the imaging equation parameters. "

"This synthetic image will have the same resolution as that of the MPRAGE and hence can replace
the acquired low resolution image. As we have no ground truth for the higher resolution T2-
w image we visually compare it with the acquired T2-w image"

##Nick
418




[GAN (Generative Adversarial Networks) NOTES] ##GANs
- Two agents with competing agents: Minimax game
  - Generator Network: Creates synthetic data from training data
  - Discriminator Network: Takes input generated data, tries to discriminate between *SYNTHETIC* and *REAL* data (by comparing against a "target" set)

- Two networks train by competing to gain the "upper hand" on the other

- Discriminator: - Binary classification
                 - Outputs probability that input data comes from real dataset
                 - Loss function:
http://blog.paperspace.com/content/images/2018/05/objective_function.jpg
                 - Desirable equillibrium: Prob = 0.5
                   - I.e.: Unsure if data is real or fake

- Generator: - Creates model w/ parameters that capture essence of images

- iu




# Log: 05.11.18
[ImageNet Classification with Deep Convolutional Neural Networks]
- Classification network employed on ImageNet images (1.2 mil HR images)
  - 60 million parameters
  - 650,000 neurons
  - Five convolutional layers, Max Pooling, Three FCL (With dropout), 1000 SoftMax

[Architecture]
- *Activation function: ReLU non-linearity*
  - Less prone to saturation (Not at all, actually)

- *Multi-GPU training*
  - Split kernels/ filters between GPUs (Exploit memory sharing abilities): E.g.: Kernels in layer 4 take only input from only kernel maps from layer 3 that reside on same GPU
  - Makes CV more complicated: But allows for tuning of amount of communication

- *Local Response Normalisation* "Brightness Normalisation"
  - Extra normalisation step to aid in generalisation
  - Normalisation over 'n' adjacent kernel maps at same position

- *Overlapping Pooling*
  - Employ s < f therefore have overlapping pooling

- *Overall Architecture*
  - Input: 224 x 224 x 3
  - Five convolutional layers
  - Max Pooling
  - Three FCL (With dropout)
  - 1000 SoftMax

- *Data Augmentation*
  - Employ label-preserving transformations (generated on CPU)
  - Image translations + reflections: Extract patches from Orig. Img. (2048 times augmentation)
  - Intensity alterations in RGB channels (Training set):
    - Perform PCA on set of RGB pixel values

[Learning]
- SGD: * BS = 128
       * Momentum = 0.9
       * Weight decay = 0.0005 (strength of regularisation)
       * Weight initialisation: Zero-mean Gaussian distribution
       * Bias initialisation: 1 (To accelerate early learning)

# Log: 08.11.18
- Research Log Skills section <N>
  - Fill in last meeting notes <N>
- Complete Course 4 Week 3 Coursera [Y]
- Submit GPU Grant [Y]
- Draw up rough plan for Ivana meeting [Y]
- Tensorflow... _WIP_





# Log: 09.11.18: Ivana Meeting: Things to mention
- Work done so far: Comparing simulators
- Next step: - Develop Image synthesis network that can identify MR parameters
             - Use extracted parameters in a simulator to simulate missing/ contrasts of interest
             - Work already done involving random forrests + segmentation techniques
             - Could still work with segmentations: Use a (cycle)GAN to circumvent flatness issues

- Ivana expertise:
  - Mention how would want volumes to be generated quickly: Want a physics augmentation layer in future if possible
  - Static equations work well but cannot account for certain inhomogeneities
  - Missing sequences in POSSUM also an issue: How quickly to develop?
  - Alternatives to POSSUM: Maybe something that is less accurate but can produce images more rapidly

- Future meetings: Invite Ivana over, mention how would be good to have big meeting with herself + DT + Jorge

# Meeting Notes
- According to Ivana there are no simulators (of sufficient complexity) that can solve time problem
  - Many aren't even completely open-source
- Two ways of getting around it with POSSUM:
  1. GPU Implementation (**Most viable**): 31 - 228 times speed-up
    http://www.bk.tsukuba.ac.jp/~mrlab/PDF_labo_paper/2017_JMR_BlochSolver.pdf
    Would also allow for complexity reduction (store MPMs only in hardware)
  2. Complexity reduction (less viable/ less likely to have a big impact):
    Remove unnecessary features

  - Given that current focus is on deep learning, can put this part on hold
    - Ivana mentioned trying to get funding to get someone to do GPU imp.
    - Can return to POSSUM, maybe in a year, and this might be a feature

- Bottleneck: Number of slices, scales as Z**2
  - Have to go through 256 x 256 x Z time points for every voxel

- Meet-up with Jorge: Happy to do it soon, talk to Jorge





# Log: 12.11.18
- Research Log Skills section <N>
  - Fill in last meeting notes <N>
- Tensorflow... _WIP_

**[TENSORFLOW NOTES]**
General workflow: - Define initialisation functions
                  - Define model
                  - Define cost function
                  - with tf.Session as sess:
                      <variable initialisation>
                      init = tf.global_variables_initializer()
                      sess.run(init)
                      sess.run(COST, feed_dict={ParamsforCost})

TENSOR.eval(): - Equivalent to calling _tf.get_default_session().run(t)_
                  **i.e. if want to evaluate variable before it has been initialised need to use a feed_dict**
               - Anything that goes into "eval" counts as a feed_dict
               - E.g.: accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                                  **Cast: Converts tensor to a new type**
                                  train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

tf.nn VS tf.layers: - Down to implementation preference
                    - E.g.: nn.conv2d takes in tensors as inputs
                    - E.g.: layers.conv2d takes in an integer as input (output dimension)

[VARIABLES]
- Initialise with tf.get_variable(name=, shape=[], initializer=tf. ...)
  - If want tf.Variable to have value of tf.Tensor can use initializer = tf.constant([A, B])

[VARIABLE SHARING: SCOPES]
- Implicitly wrapping tf.Variable within tf.variable_scope
  - Can do so explicitly, not a difficult task
  - Most tf.layers use this approach

- **Example**
- Create a function for a conv + relu layer
  - Will have such variable names as "weights" and "biases" in this function
  - *Cannot realistically call this function more than once*: Will fail because TF won't know whether to re-use them or create new ones (from the first call of it)
- Can get around this problem by using scopes:

def my_image_filter(input_img):
   <with tf.variable_scope('conv1'):>
     relu1 = conv_relu(INPUTS)
   <with tf.variable_scope('conv2'):>
     return conv_relu(relu1, OTHER_INPUTS)>

What if you want variables to be shared? - Call scope with same name + reuse
                                           *tf.var_sco("NAME", reuse=True)*
                                         - Call [scope.reuse_variables()]
                                           *Do this in between function calls*
                                         - Initialise scope based on another one *using with*

[COST COMPUTATION]
- Don't need to call SOFTMAX in final FCL, bundled with cost:
  - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y)
[REGULARISATION]
Dropout: - Apply (only?) after tf...fully_connected w/ keep_prob
         - After applying dropout, apply another FCL

L2-loss: - Define regulariser: <tf.nn.l2_loss(LAST LAYER)>
         - "Add" to standard cost: <COST + lambda * regulariser>
           *Cost will be some softmax cross entropy, for example*

[MODEL PROCESS]
Initialise placeholders (X, Y, weights, constants)
Instantiate <FP>, <COST>, <OPTIMIZER>
Call session
...

- Have now [PARAMETERS] saved
  - Accuracy: Comparison between predictions & GT
  - Predictions: Need to run 'X' through the fit *w/ saved params*
    - Involves calling FP to get Z3
- When calling "accuracy", need to pass all necessary variables to feed_dict:
  - <X (required by FP)> <Y (required for comparison)>
  - <KeepProb (required for FP)> <Lambda (required for cost)>

[L2 CONSTRAINED SOFTMAX]
- SoftMax Loss: * Widespread use in DCNN
                * Biased to sample distribution
                * L2 Norm == measure of quality (proportional)

  + [Easily implemented]
  + [Any input batch size (vs triplet loss)]
  + [Converges quickly]
  # Biased against bad examples (maximise conditional probs.)
  # No verification requirment of keeping +ve pairs close, -ve pairs far apart *Metric learning + SoftMax features* OR *Auxilliary loss + SoftMax loss*

- L2 SoftMax Loss: * Constraint on features (DURING TRAINING) so that
                     L2 norm remains constant => Constant radius HS

  + [Similar attention for good + bad examples]
  + [Strengthens verification signal: Similar features close, far features far] *Maximise margin for normalised L2 distance*

**Implementation** ZN = tf.divide(ZN, tf.norm(ZN, ord='euclidean'))





# Jorge Meeting: Notes
- Future of MRI simulations:
  - Want to ultimately add a physical augmentation layer to NiftyNet
  - Won't be possible with something like POSSUM: Would need to add FSL as a dependency (Even w/ GPU support)
- Insight into TF functions: e.g.: tf.nn.softmax_cross_entropy_with_logits()





# Log: 15.11.18
- Research Log Skills section <N>
  - Fill in last meeting notes <N>
- Write up initial network plan
- Tensorflow... _WIP_ (See notes above): * L2 contrained SoftMax
                                         * Model process
                                         * See: colab for continued work on MNIST (+ Coursera Signs) classification:
                                         * https://colab.research.google.com/drive/1ulJ95ZFhwyMmwyXRXHaeOs0vjrZ8gSHw#scrollTo=OF0sdG0KabPQ
                                         * See: <C:\Users\pedro\Desktop\Project\ML\Tensorflow Intro>

**[REGRESSION MODEL]**
Motivation: - Want to be able to synthesise missing modality images of subjects given at least ONE image

Theory: - Learn regression (mapping) between images of <modalityA> and <modalityB>
        - Can apply SAME regression to another subject's image of <modalityA> to get <modalityB>

Methodology: - Summary: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4532609/pdf/nihms692268.pdf (Figure 1)
             - Have _Classification Network_ to estimate pulse sequence parameters (@Params) of given image [Image S_a]
             - Apply @Params to Atlas of qMAPS to synthesise Atlas image of same modality [Image A_a]
             - Have _Regression Network_ to find mapping between [Image A_a] and Atlas image of DESIRED modality [Image A_b]
             - Apply LEARNED REGRESSION to [Image S_a] to synthesise [Image S_b]


**[DATA]**
Atlas construction: - 27 subjects (patients + control), early onset Alzheimer's Disease (AD) dataset
                      - qMPM for every subject
                      - Assemble qMPM into one large Atlas *qMPM Atlas*
                        - **JOG has Atlas of a SINGLE subject (hold-out) and tests on ALL other subjects**
                      - Synthesise images to create "simulated" equivalent modality Atlas *Mod. Atlas*
                      - Could also use my acquisitions for this purpose


**[NETWORK DEVELOPMENT]** @Written on 17th
Regression leads: - HighRes3DNet: https://arxiv.org/pdf/1707.01992.pdf
                  - Deep Regression Neural Networks: https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33






# Log: 16.11.18
Paper link reminder: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4532609/pdf/nihms692268.pdf

@MIMECS **CHECK**
- Magnetic Resonance Image Example-Based Contrast Synthesis
- Also a patch based method:
  - Build Atlas of "SINGLE" patch from multiple images (Multi-modality)
  - Choose "SEPARATE" patch from subject images (Multi-modality)
  - Learn mapping between Subject and Atlas patches
  - Synthesise image using learned coefficients based on "ANOTHER" Atlas (Physically related to original Atlas)


@D-Syn **CHECK**
- Symmetric diffeomorphic image registration with cross-correlation
  - Register A1 to S1
  - Apply same deformation to A2 -> S2
  - No learning carried out for mapping, _purely a registration_ approach
  - (Compared to learning the mapping between Atlas synthesised A1 and desired Atlas contrast A2 => applying mapping from S1 to get S2)


@JOG Parameter estimation method **CHECK**
- Use of fuzzy c-means to estimate means of main tissue types: *Requires tissue segmentations*
  - CSF, WM, GM
  - Assume that mean intensities due to MEAN TISSUE PARAMETER VALUES
  - i.e.: [Mean Tissue Intensity] = ImagingEqn([Mean qParams], <SeqParams>) *Times three for all Tissue Types*
    - [Mean Tissue Intensity] + [Mean qParams] known
    - <SeqParams> unknown
  - Will typically have 4 unknowns in <SeqParams> and 3 eqns (MTS)
    - Use Newton's method to solve (Assume knowledge about one <SeqParam>, e.g.: TR by looking at header)


@JOG Regression method (contrast learning) **CHECK**
- Have synthesised [Image A_a]
- Want to learn intensity transformation to convert [Image A_a] to target image [Image A_b]
- Use _Non-linear regression_: * Consider image patches of [Image A_a] together with CORRESPONDING central voxel intensities in [Image A_b]
  - Pick (n x n x n) patch centred on voxel v: Flatten and use as dependant variable: Ensure spatial smoothness
  - Apply random forest regression to learn mapping relationship


@JOG Methodology: How to evaluate model, Atlas/ Synthesis  subject split etc. **CHECK** ##STEPS
  1. Estimate Pulse Sequence Params of [Image S_a]
  2. Synthesise [Image A_a] using Params + qMAPs from Atlas
  3. Learn transform between [Image A_a] & [Image A_b]
  4. Apply same regression to [Image S_a] to obtain [Image S_b]

- Choose one subject for ATLAS
  - Does not matter if legions are present, these are T1W + T2W blind
  - Remainder 20 subjects co-register to MPRAGE
  - Compare synthesised image to T2W image


**[Initial Tests]**
- Can resort to BrainWeb Digital Phantom for MPM + Modality volumes for first batch of tests: http://brainweb.bic.mni.mcgill.ca/
- Normal brain database --> Modality choice + params --> Download
  - Downloads as minc file: Wrote script to convert from minc to nifty file for ease of use <C:\Anaconda\Lib\minc_to_nifty.py> *Can easily import in Python: Eponymous function for conversion*
  - Further information on MINC data format: http://brainweb.bic.mni.mcgill.ca/about_data_formats.html
- JOG paper builds MPMs using curve fitting by downloading multiple images of a given modality: T1 from SPGR, T2 from DSE, PD from both


**[Network Choice + development]**
# See: Log 16.11.18
Summary:
+ _Deep Regression Neural Networks_: https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
+ HighRes3DNet: https://arxiv.org/pdf/1707.01992.pdf
  + Check NiftyNet networks






# Log: 19.11.18
- Things to go through: * Network choices
                        * Standardisation procedures: HOW?
                        * Kerstin paper (Deep Boosted Regression)
                        * Tensorboard: Experimentation

[STANDARDISATION]
- Intensity standardisation: No consistent image intensity scale for diff. tissues: Not quantitative
  - No problem for diagnostics
  - Problem for image processing algorithms
  - Difficult to compare MR data from different sites/ even intra-subject scans

- How does Jog's paper tackled this (Using PSI-Clone)?
- Aim: Standardise BrainWeb SPGR to atlas target SPGR
  - Simulated various SUBJECT images by varying PS params.
  - Have single subject Atlas
  - Carry out [Steps 3 + 4] *Regression + transform*
  - Subject + Atlas same modality -> **STANDARDISATION**

[KERSTIN PAPER]
Aim: Synthesising CT images from MR acquisitions

- Regression method: *Check boosting models*
  - Find mapping between T1 + T2 --> CT
  - Highly non-linear: Composition of N simpler functions w/ N params
    - Learn Params via Loss FN minimisation
    - Optimisation + computation problem: Curse of dimensionality (3D)
    - Recursive parameter optimisation: Every "corrective" function acts on previous function
    - Overfit from parameter excess? Share Params. btwn functions f2+ *Function f1 makes first synthesis mapping approximation at iteration i=0*
    - Functions f2 --> fN == iterations 2 --> N, iterating over mapping problem, improving approximation with every iteration
      - i.e.: Function fn maps prediction f(i-1) w/ Params NC to a better approx.
  - Loss function minimises @Sum(yk - y)**2

- Network architecture: https://arxiv.org/pdf/1707.01992.pdf
  - Two separate CNNs <Why>
  - Need efficient 3D learning represenation capabilities from large-scale data: **High Res. Compact**
    - [20 ConvLayers]
      - *First seven*: 16 (3 x 3 x 3) kernels (low level features)
      - *Next twelve*: 32 - 64 (3 x 3 x 3) kernels (Mid-HL features)
      - *Last layer*: 160 (1 x 1 x 1) kernels (SoftMax)
    - Length two residual connections (Identity map + efficient)
  - Feed output of 1st Network (pCT) to 2nd Network (pCT-CT residuals)
    - Iterate over second step, adding residuals to pCT until conv.

[TENSORBOARD] ##TENSORBOARD
Need to explicitly set which variables are to be tracked in script

1. Use <summary>: tf.summary.histogram("VARIABLE_NAME", VARIABLE)
2. Use <FileWriter>: train_writer = tf.summary.FileWriter('./logs/n/train', sess.graph) *During training: Right after sess(init)*
   2a. <mergeAll>: merge = tf.summary.merge_all() *During epoch loops*
        Ensure that summary is calculated when sess is run
   2b. <AddSumm>: train_writer.add_summary(summary, counter)
        *Set up counter during loop*
3. Run tensorboard (in CL): tensorboard --logdir logs/1 *Wait a minute after running*
   3a. Go to URL: http://desktop-mok8hgt:6006

> Other Options
--samples_per_plugin images=N *Sets how many images visible on slider*
--port N *Set display port*
--logdir=[A][DIR_A],[B][DIR_B]... *View multiple log file outputs* => See manyboard.sh (Automates this process)

> Running from DGX1  ##DGXBOARD
1. ssh -N -f -L localhost:<LOCALPORT>:localhost:<DGXPORT> pedro@172.29.29.30 *On local machine*
2. tensorboard --logdir logs/ --port <DGXPORT> *On DGX1*
3. Navigate to http://localhost:<LOCALPORT> *On local machine*

- To close ssh connection (To prevent errors/ free up ports): 1. netstat -tulpn *List all open ssh ports*   ##PORTS
                                                              2. Find ssh process with open connection (look for desired port)
                                                              3. kill <ProcessID>



# Log: 20.11.18
- Install NiftyNet on cluster
- Try demo example
- Create some T1/ T2 images to test on HighRes3DNet

[NIFTYNET]
git clone https://github.com/NifTK/NiftyNet.git
Help pages: * https://niftynet.readthedocs.io/en/latest/
            * https://niftynet.readthedocs.io/en/latest/config_spec.html

Running from the cluster: * See <submissionFile.sh>
                          * See <fernando.ini> *Config file*

Configuration Notes:
- Spatial window size: Size of "windows" of volume






# Jorge Meeting notes:
- First approach: - Train (2D!) network to map between T1 images (2D! + choose ALL subjects, not just healthy/ diseased)
                    - Simulate two sets of T1 images: Different flip angles
                    - Choose, say, 10 middle slices of each subject
                    - Learn regression between datasets
                      *HighRes2DNet?*
                    - No need for augmentation at first approach
                    - Standardisation mention <N>

- Learning discussion: - WIP: Many different ways to approach DL parameter optimisations
                         - Curriculum learning (focus on hard areas), teacher/ subject learning etc. *NVIDIA self-driving cars*
                         - Very much a "hands-on" field, need to try things out to gain intuition
                         - Also important to know your problem: Save a lot of time if have some intuition of where to look

- Data requirements: **Scales according to cardinality of output VS input**
                     - Classification: I/O size >> 1, a lot of condensed information --> Need far more data
                       *E.g.: Cat in image?*
                     - Mapping: I/O size ~ 1, 1-to-1 --> Need much less data
                       *E.g.: T1-w to T2-w mapping*
                     - Makes regression problem far more feasible given ~ 20 subjects
                     - When doing parameter extraction will need far more data (Output ~ 6), but can simulate it






# Log: 21.11.18
- Keep working on cluster submission of NiftyNet tutorial <N>
- Update Research log meeting notes <N>
  - Update "goals" <N>
- HighRes2DNet: Work on synthesising appropriate training/ testing sets <N>

[UNet Demo]
- Tutorial found at: http://www.miccai.org/edu/finalists/U-Net_Demo.html
- Downloaded data from: http://celltrackingchallenge.net/2d-datasets/
- Working directory: <C:\Users\pedro\Desktop\Project\ML\NiftyNet Demo>
- Working directory on cluster: </home/pborges/UNet>
  - Git cloned NityNet into </home/pborges/NiftyNet>
  - Submission file: [submissionFile.sh]
**ERROR** - Had problem with libculs import (from tensorflow)
            - Had to change Cuda version in SF from 8.0 to 9.0
- Unzip files into </home/pborges/UNetDemo/data/u-net>
  - Copy [file_sorter.py] + [make_cell_weights.py] from </home/pborges/NiftyNet/demos/unet>
  - Run [file_sorter.py] followed by [make_cell_weights.py]
  - *Call BOTH with argument! --file_dir*
  **ERROR** - Had problems with missing "demos.unet.file_sorter" module
              - Changed line to call just file_sorter (in same folder so works)

[Python on the cluster]
- On DESKTOP downloaded Linux version of Anaconda from: https://www.anaconda.com/download/#linux
- Transferred to cluster
- Run: <bash ANACONDAFILE.sh>


[HighRes2DNet]
- No 2D implementation: Need to go through HighRes3DNet imp. and check for compatibility
- **[NOTE]** Full_seg_and_mpm_reg.py for ##ONELVLALADIN
- **Need to watch out with registration: Aladin takes FILENAMES, NOT ARRAYS**






# Log: 22.11.18
- Keep working on cluster submission of NiftyNet tutorial _WIP_
- Update Research log meeting notes [Y]
  - Update "goals" [Y] *Still need to flesh out descriptions*
- HighRes2DNet: Work on synthesising appropriate training/ testing sets <N>

[NiftyNet tutorial]
**Problems with installing Tensorflow**
- Previously erroneously installed MAC version <NOT> Linux
- Created new environment: <conda create --name tensorflow python=3.5>
  - <conda activate tensorflow>, <conda deactivate>
- pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp35-cp35m-linux_x86_64.whl *Version 1.10! Python 3.5, GPU*
- Will want to try (outside environment): https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl *Version 1.12 Python 3.6, GPU*
- <N> Does not seem to work in standard environment **Because it's Python 3.7!**






# Log: 23.11.18
- **Cancelled** NiftyNet tutorial
  - Instead try to implement one of the Model Zoos *dense_vnet_abdominal_ct_model_zoo*
  - https://niftynet.readthedocs.io/en/dev/model_zoo.html [Y]
- HighRes2DNet: Synthesise all MPMs + T1s [Y]
- Sign up for Jade-14 project on Jade cluster <N> *No connection can be made for some reason*
  - https://um.hartree.stfc.ac.uk/hartree/login.jsp


**[Model Zoo tutorial]**
- Set up submission file (similar to before): </home/pborges/modelZoo.sh>
  - Same variables as before, calling python from </share/apps>
  - Call net_download.py to get abdominal dataset
    <python net_download.py dense_vnet_abdominal_ct_model_zoo>
  - Call net_segment.py afterwards to get segmentation:
    <python net_segment.py inference -c /home/pborges/niftynet/extensions/dense_vnet_abdominal_ct/config.ini>

- Working directory: </home/pborges/niftynet> + <models> + <data> + <extensions>
- Input: <data> subdirectory *100_CT.nii*
- Output (segmentation): <models> subdirectory *segmentation_output/100__niftynet_out.nii.gz*


**[MPM Creation]**
**NOTE** **R1 and R2s maps are in unknown units! See below**
- Working in directory: <C:\Users\pedro\Desktop\Project\PhD\AD Dataset>
- Revisit T2s creation process: Rectified linear function to prevent stark intra-region contrasts *See: ##RECTIFIED*
- Using SPGR to generate T1 images of two different flip angles for mapping learning: ##T1SIMS
  - TR = 18ms
  - TE = 10ms
  - FA_1 = 15 deg.
  - FA_2 = 60 deg.
  <MISTAKE> Very large values observed in final images, units likely mismatched somehow:
    * R1 maps: Values ~ High 100s/ Low 1000s. Expected for T1 (ms) NOT R1
    * R2s maps: Values ~ 0.01. Expected for R2s in ms^-1
    *See MRI book for reference value table*

- Need to get code for rectified linear function for proper T2s creation [DONE: In C:\Users\pedro\Desktop\Project\DT MPM section]

- Corrective function: - **SOFTPLUS: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus**
                                                         <log(1 + Ae^Bx) + c>
                                                         * [A=10]
                                                         * [B=0.3]
                                                         * [C=4]
                    **Values need to be fine tuned further (very little difference right now, 0.1 vs 100s)**

- Correction procedure: 1. Make a mask of R2s of negative values (1 == neg, 0 == otherwise)
                        2. Apply ReLF to ORIGINAL R2s, multiply by mask (So positive values unaffected by correction) *ReLF R2s*
                        3. Make a copy of R2s where all negative values are zero *Zero R2s*
                        *This is done to ensure that the correction can be added onto a "clean slate"*
                        4. Add ReLF R2s to Zero R2s to obtain final, corrected, map *R2s Corr*

Next time: * Config file setup for network
           * Jade account setup







# Log: 26.11.18
- Try to run HighRes2DNet with **MY** generated maps
  - Need to generate these maps, same params as before: ##T1SIMS
    - Maybe vary TR and TE slightly to add variability to inputs

MPM datasets: * Contrast is correct: WM > GM (R1), DGM > WM (R2s)
              * Units seem odd, DT will get back on this

[HighRes2DNet]
- Configuration file for regression: Specify "volumes" as (H, W) instead of (H, W, 1)
  - Exception: Volume padding (obvs): (HP, WP, 0)
- Example config file: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/unet-demo/config/default_monomodal_regression.ini
- Window size notes: https://niftynet.readthedocs.io/en/dev/window_sizes.html

- Created config file: </home/pborges/standardisation.ini>
- Created directories: * Input: <T1_FA1>
                       * Target: <T1_FA2>
                       * Model: </home/pborges/standardisation>

- [loss_border] : * Used to crop 2* lb pixels from input image
                  * Don't want to learn from regions of little information (i.e.: background)
                  * Cropping of sides 'eliminates' background

- [Downsampling] : * Downsampling with layer
                   * Or with nibabel
                   * Concerns with too many parameters, "less meaning per"

watch qstat
tail -f JOBID

_Application_
- When running NiftyNet, need to specify application:
                                  [python -u net_run -c <CONFIG> -a <APPLICATION>]
- <APPLICATION> specifies what type of task is being carried out
  - Segmentation: niftynet.application.segmentation_application.SegmentationApplication *Alias: net_segment*
  - Regression: niftynet.application.segmentation_application.RegressionApplication *Alias: net_regress*
  - Autoencoder: niftynet.application.segmentation_application.AutoencoderApplication *Alias: net_autoencoder*
  - GAN: niftynet.application.segmentation_application.GANApplication *Alias: net_gan*

- Alias allows you to EXCLUDE the -a option since it is specified at the <-u OPTION> stage

_DATA_
- Intensities of FA = 15 vs FA = 60 seem extreme *Over 4x difference*
  - Opt for FA = 15 vs FA = 45 *~2/3 x difference*

- Slice creation: <C:\Users\pedro\Desktop\Project\PhD\ad_t1_syn.py>
  - 20 slices
- **NoTE**: * Because taking slices need to adjust affine transformation when saving
                      [AFF(2, 3) -> AFF(2, 3) + z_midpoint/2]
            * This will ensure the slices are centered relative to volume (in Z)






# Log: 27.11.18
- Keep working on own data regression task (FA)
  - Data status

[HighRes2DNet]
- Had output: </home/pborges/standardisation>: * dataset_split.csv *Specifies which files are used for training/validation/inference*
                                               * MODALITY1.csv *Input images*
                                               * REGRESSTARGET.csv *Regression targets*
                                               * settings_training.txt *Config file parameters*
                                               * training_niftynet_log *Log of training*
                                               * <logs> **EMPTY**
                                               * <models> **EMPTY**
  - <models> + <logs> directories shouldn't be empty
    - Found that image sizes were incorrect: (181, 217), config had (256, 256)
    - Corrected .ini by changing spatial window sizes *Don't forget to change inference as well!*
    - Ran again:
      - **ERROR**: KeyError: 'image'
      - Key errors associated with config file mistakes
      - In this case, ommitted "image" parameter under [REGRESSION]






# Log: 28.11.18
- Keep working on HighRes2DNet for own images
  - Data status

[MEETINGS]
- Dave Thomas wants to have a meeting to discuss PS simulations w/ Irtiza Gilani + MR protocol optimisation/ harmonisation
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5045042/ *dMRI Harmonisation across sites + scanners*
  - https://www.biorxiv.org/content/early/2018/05/04/314179 *Retrospective Harmonisation: Multi-site dMRI data*

[HighRes2DNet: Or is it?]
- HighRes3DNet not compatible with 2D images: * Uses 3D kernels
                                              * Could change HighRes3DNet to allow for 2D image acceptance
                                                * Not worth the effort: Likely to introduce bugs/ take a lot of time
                                              * Instead, opt for **U-NET (2D)**
                                              * [name=unet2d]
                                                * https://arxiv.org/pdf/1505.04597.pdf
                                              * Image becomes smaller: Changed <unet_2d.py> from [VALID] TO [SAME] padding
                                              * **ERROR**: When up-conv, tensors the same shape so resulting difference < 0
                                                * Not sure how to tackle this, may be better off just trying 3D

[BASH ASIDES]
- Creating alias to allow for new command: 1. Create bash script
                                           2. In <bashrc> create alias: <alias COMMAND='bash SCRIPT.sh'>
                                           3. source <bashrc>
                                           4. Can call COMMAND instead of COMMAND.sh

[DATA STATUS]
- Dave Thomas not sure yet (waiting on Alex), but educated guess is: * R1Map: 1000 x s^-1
                                                                     * R2sMap: ms^-1
  - Can work with this for now to get some simulations *Work in seconds*

# Jorge meeting: Notes
- Helped debug error: Missing option under [REGRESSION] in config
- General discussion on static vs dynamic graphs: Usefulness of each dependant on application: E.g.: Dynamic are well suited for natural language processing because don't know how long a sentence if going to be
- Idea of "transformation placeholders" in networks: Since networks can be used for different applications (segmentation, classfication, regression), last layer's activation function will change, accordingly: E.g.: SoftMax for Classification.
  - Therefore have placeholder for function in network: In case of regression, since it needs to be "set", use identity mapping (no change)






# Log: 29.11.18
- Retry HighRes3DNet: Use 3D windows [Y]
  - Need to generate volumes, accordingly
- Respond to DT email about meeting <N>
  - Sort out "overview"

[HighRes3DNet]
- Synthesised volumes for all 27 subjects for two FAs
- Moved (temporarily) to </home/pborges/standardisation/Volumes>
  - Followed by: * </home/pborges/Volumes_FA_15>
                 * </home/pborges/Volumes_FA_45>

[UNet_2d]
- Re-trying UNet_2d
  - Commented out concat. layers: Should not be necessary with 'same' padding
- WD: </home/pborges/unet_standardisation>
- ini: </home/pborges/unet_standardisation.ini>
- sh: </home/pborges/qsub_unet.sh>

**NOTE** Strings in .ini file should **NOT** be surrounded by apostrophes/ quotation marks


- **TENSORBOARD**: * Currently don't have an easy way to access directly from the cluster
                  * Tried similar approach to calling python in qsub sh files, no success </home/pborges/tb.sh>
                  * Pritesh approach: Download log files and run locally *Cumbersome*
                  *

[VAEs vs GAN]
GANs: * Generated off arbitrary noise
        * If want specific features need to search over entire distribution
      * Only discriminates between _REAL_ and _FAKE_ images
        * No constraints that image "needs" to look like the object
sem. sch.
VAEs: * Solves these problems
      * _TBC_






# Log: 30.11.18
- Run inference on Unet_2D + HR3DNet <N>
- Respond to DT email <N>
- Continue reading about VAEs/ GANs _WIP_

[Models]
- Tensorflow saves three different files: * <meta> Graph structure
                                          * <ckpt> Checkpoint
                                          * <data> Values of each variable in graph
                                          * <index> Identifies the index of the checkpoint

- Currently running inference on lower (2000/10000) model checkpoint for HRN + UNet_2D

[UNet-2D] ##BATCH
- Should be using larger batch sizes: * [m < 2000]
                                      * [Lower noise]
                                      * [High GD Steps]
                                      * <LongerCompTimes>
  - Started using BS = 10

[HRN: T1W - T2W]
- Synthesised 27 T2W DSE volumes
  - *Corrected CSF zeros still seem "dark", have another look*
- </home/pborges/HRN_T1_T2>
  - BS = 32 (Test)
- sample_per_volume == samples taken from **SAME IMAGE**


# Log: 03.12.18
- 1D conv: When expect interesting features to arise from short subsequences pf input
  - Feature space reduction (3rd dimension)
- Running HRN_T1_T2 (16 samps) + HRN_T1_T2_b1 (8 samps) again w/ lower interp order
  - **NOTE** b1 will overwrite models, probably

[Inference]
- "Blocky" outputs observed: Why?
  - Seem to be same size as spatial window
  - Why learning according to spatial window size??


# Log: 04.12.18
- Interp. order:
  - Main differences arise when using low interp across the board
  - Maybe write for loop for many different number of volume samples

Inference test ideas:
  - Can investigate performing inference at different times
    - Until:
until [  $VALUE -lt N ] *Iterate until $VALUE is less than N*
    do

    let max_iter_val = VALUE - STEP
done

[UNet 2D images]
- No inference/ proper training yet: Need to correct [REGRESSION] in .ini file

# Jorge meeting: Notes
questions: - Can traditional machine learning approach be used to inform decision of network type?
           - Should we use a residuals approach for more complicated regressions?
             - Force no negatives?
           - Problems w/ UNet 2D: Padding/ Concat method

Next steps:
  - T1 -> T1 maps of DIFFERENT sequence types
    - MPRAGE -> SPGR
    - Keep params the same! (intra-sequence)
  - Many T1 -> T1 maps of DIFFERENT sequence types
    - 4 MPRAGE -> 1 SPGR
    - Can mess around with data augmentation
  - Ultimately want to learn invariance between N images of a subject to a standard
    - Can take the network representation (i.e. penultimate activation, w/out image creation) since this representation should be invariant to types of input sequences

- Valid vs same padding:
  - [VALID] means that only the "meaningful" parts of the data will be considered BUT results in diminished input size
  - [SAME] allows input/ output to have the same dimension but patch borders will have information loss (because learning from padded sections, i.e. 0s)

[MPRAGE]
- Need to figure out how TR + TE factor into imaging equation
  - Equation params are only: TI, tau, TD: Where do TR and TE come in?

Mounting comic: *sshfc*






# Log: 05.12.18
[MPRAGE (3D)] **NOTE** Can clarify with DT
- Only one inversion
- One TR for each slice
- One echo (?) for each slice (This is the case if TE ~ 0.5 TR)

[Parameter estimation]
- Assume that intensity at any voxel due to underlying tissue params.
- Can use simplified imaging equations to describe this relationship
  - It follows that the average intensity of a class is due to average tissue params.
  - Therefore obtain set of 3 (Tiss. Types) eqns + 4 unknowns (G, TI, tau, TD)
    - Solve using Newton's method: https://math.stackexchange.com/questions/268991/how-to-solve-simultaneous-equations-using-newton-raphsons-method
    - THEREFORE don't actually need to know sequence params used (Maybe one: TR)


# Dave Thomas meeting
Overview
- qMPM approach of: * _Static equations_
                      [+] MPRAGE, DSE, SPGR
                      [+] Quick, easy to implement (Turing machine)
                      <-> Not versatile, not as accurate (approximation)
                    * __Full Bloch Equations (POSSUM)__
                      [+] Bloch equation solved for each object, tracking MV through time
                      [+] GRE sequences
                      [+] Diffusion-weighted sequences
                      [+] Noise modelling
                      [+] Spin history: Motion, B0 inhomo
                      [+] Artifacts: Susc., CS
                      <-> Slow: Could benefit from GPU accel.
                      <-> No SE Sequences, MPRAGE (SE better for inhomo, susc, CS; GRE faster)
                      <-> Bulky software requirements: FSL

Current work
- Scanner/ sequence standardisation: * Using Deep Learning + static equations to try to achieve invariance
                                       * Don't want to try to tweak scanner params. to get == image from another scanner
                                       * Want to STANDARDISE to single type of image

MPRAGE question:
- Have simplified MPRAGE equation
  - Involves TI (~700ms), TD (~300ms), tau (~10 ms): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4039442/pdf/pone.0096899.pdf (MPRAGE equation)
  - BUT paper only mentions TI: Presumably can synthesise using just TR, TE, TI?
  - How to relate?
  - Even if only TR is used, where does it feature in equation? * [TR = TI + N*tau + TD] ~ Deichmann paper
                                                                * BUT TRs in JOG paper seem to be much lower

[Samps]
Increase proved to be no better than standard


# Log: 6.12.18 + 7.12.18
- MPRAGE papers: * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4039442/pdf/pone.0096899.pdf (MPRAGE equation) (MPRAGE Params) ##MPRAGEPARAMS
                 * https://tinyurl.com/y8oqjnpu (Deichmann: 3D MPRAGE Optimisation)

- SPGR papers: * Just look at JOG

- Consider using 'VALID' instead of 'SAME' w/ HRN






# Log: 10.12.18
- Running MPRAGE -> SPGR </home/pborges/MPRAGE_SPGR>
          MPRAGE -> SPGR </home/pborges/MPRAGE_SPGR_AUG> *With augmentations*
- MPRAGE Equation problem: * Paper was **WRONG**: TR is meant to be TI
                             * Makes sense: No TR in equation, need to give algorithm at least one param to solve with Newton's method
- *FUTURE SOFTWARE*: https://mrtrix.readthedocs.io/en/latest/
- Check Jade account

- Pending questions: *

[SE vs GRE]
- http://mriquestions.com/spin-echo1.html (SE specifically)
- http://mriquestions.com/gre-vs-se.html (GRE vs SE)

[NII headers]
https://www.nitrc.org/docman/view.php/26/204/TheNIfTI1Format2004.pdf
- dim0: Specifies the dimensionality of the volume
        * E.g.: dim0 = 3 == 3D volume
                dim0 = 4 == 4D volume
- dim3: "Time" axis
- dim4: Values to be stored at each spatio-temporal location
        *1 if scalar. > 1 if vector*

[SKULL STRIPPING]
Can use BET/ BETSURF
  Or NiftySeg
  Atlas-based approach: <MONSTR> https://www.nitrc.org/projects/monstr

[UNET]
- **PROBLEM** Outputs don't match 884736 != 512
  - Because 96^3 != 8^3
  - See example: unet_config.ini https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/e791a7bd7ccda5e02f0650367c34263baaee872b/config/unet_config.ini

[WEIGHTED SAMPLING]
- See MR to CT example: https://github.com/NifTK/NiftyNetModelZoo/tree/5-reorganising-with-lfs/mr_ct_regression
- https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/mr_ct_regression_model_zoo.md

[Medical Image Synthesis for Data Augmentation and Anonymization using Generative Adversarial Networks]
- Investigate skull stripping
- Use bet






# Log: 11.12.18
@Name: ppb27-mxm01@dgj419
@Password: Unneeded: Public Keys
[SLURM SUMMARY] * <Partitions> Logical unit that breaks up cluster into useable units depending on qualities/ traits of different nodes
                * <Accounts> Allocated: [USERNAME]
                             Owner nodes: Same as partition name
                             Owner-guest: Guests

**USE** --account=ppb27-mxm01@dgj419 --partition=long/small

[COMMANDS] * squeue *List of all running jobs (not user specific)*
            * squeue -u <USER> *Jobs running for that user*
            * squeue --account=owner-guest *Jobs for current account*
            * squeue --partition=<PARTITION> *Jobs for specific part.*

            * sinfo *Status of different partitions*
                    *Alloc, maint, drain (unused), down, idle (Available nodes)*

  \\alias\\ * si/si2: Like sinfo, more info
                      Socket count | Core count | Thread count
                      NODES Alloc | Idle | ??? | ???
  \\alias\\ * sq: More detailed info. on running jobs
                  Priorities/ Account names/ Groups etc.

[LAUNCHING] * srun <SCRIPTNAME> *Launch interactive jobs*
            * sbatch <SCRIPTNAME> *.slurm extension*
              * Job states: PD (Pending), R (Running), CG (Cancelling)
            * scancel <JOBID> *Cancel job*
            * scontrol show job <JOBID> *More detailed job info*

[SLURM BATCH SCRIPTING] 1. SBATCH directives *Scheduler info, mem reqs*
                        2. Environment setup
                        3. Scratch setup
                        4. Copy data to scratch (Heavy IO loads)
                        5. Run your job (MPI (parallel) if need be)
                        6. Copy data from scratch

**BASIC SCRIPT REQUIREMENTS**
> Sbatch specs
#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --account=owner-guest
#SBATCH --partition=<PARTITION>
#SBATCH -o qe-%j *Where to output files* Use JOB Number with %j

> Environment specs
NAME=<NAME> *Input file*
DATADIR=<DATASTOREDIR>
SCRATCH=<SCRATCHDIR>


# Highway networks
# CERN data analysis invertible networks

# Jorge Meeting: Notes
> Project Plan for forseeable future
Know that mapping from Protocol 1 -> Some protocol 2 should be "easy"
  This is the gold standard

However, when a network tries to train with multiple protocols as input (simulating acquisitions from different sites with slight parameter variations) -> Some protocol 2 it should not perform well
  This is because it is extremely hard to distinguish between images of similar contrasts with differing parameters (Even for a human)

How to circumvent this issue: Feed physics parameters into network paired with corresponding image -> Should provide enough context to network to learn mappings: Know the manifolds of physics parameters

First approach: * Alter HR3DN to add more residual connections
                  * See Highway networks
                  * "All the way" connections, more paired connections
                  * In the ambit of improving identity mapping

**NOTE** Previous work: * Incorrect
                        * Need for each subject to randomly pick a protocol N, not have all N protocols for all subjects





# Log: 13.12.18
[HighWay Networks]
- Type of network that enables implementation of deeper networks without encountering problem of vanishing gradients
  *Remember: Deeper networks have worse performance because of vanishing gradients, adding more layers alone cannot decrease representational power*
- VS residuals: HighWay more suited for language processing/ translation; ResNets more suited for computer vision applications

[Dilated Convolutions (Upscaling filters)] (As seen in HR3DN)
https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/ (General discussion of Dilated Convolutions)
https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807 (Specific discussion of receptive fields)
- "Vanilla" convolutions struggle to integrate global context
  - Receptive field == l*(k-1)+r _l is layer index, r is previous layer's feature's receptive field size_
  - Dilation factors: Spacing between kernel elements in kernel

[HighRes3DNet: Construction]
> Layers
1. Convolutional layer (3 x 3 x 3 kernels)
2 - 4. ResBlocks (3 x 3 x 3 kernels) Dilation = 1
5 - 7. ResBlocks (3 x 3 x 3 kernels) Dilation = 2
8 - 10. ResBlocks (3 x 3 x 3 kernels) Dilation = 4
11. SoftMax/ (1 x 1 x 1 kernels)

> BNLayer: Batch normalisation
- Regularizers: Trainable params. Beta (0) + gamma (1)

> DilatedTensor
- Takes previous layer, dilation factor as input
- Iterate over however many repeats there are by creating an instance of <HighResBlock> class, then calling instance with current tensor (which calls the layer_op method in <HighResBlock>)






# Log: 14.12.18
- _Fill in Dave Thomas meeting notes_
* Irtiza developing "mid-range" MRI simulator
  * Instead of calculating Magnetisation at every time point, only update vector at every event
    * e.g. in MPRAGE: 1. Update at Inversion
                      2. Update at flip angle one
                      3. Update at flip angle two
                      4. ...
    * One-voxel Fourier Transform to convert from k-space to real space
> Issues
- Slow: 5 hours + for a slice
  - Possibly due to to excess calculations past the steady state
    *i.e. don't need to keep updating magnetisation if it has reached a steady state and is unlikely to change*
  - Possibly the bottleneck lies in the FT: In this case then the previous "issue" becomes less important, but worth correcting for





# Log: 17.12.18
- Keep working on modifying HR3DN
  - Add extra residual connections (over-arching)

[HighRes3DNet: ResExtra]
> HighResBlock
- Loops over two layers
  - Final step: Create residual connection by carrying out elementwise connection between initial tensor and final tensor

> Altering to create new connections
- Want to have over-arching connection between first and final layer of each "block-size"
  - Save input tensor, calculate paired residuals (as before), then "connect" at the end w/ initial tensor

> Testing
- Changed only for first residual "triplet"
- Need to continue for remaining "triplets"

> Calling altered HR3DN in notes:
- name=extensions.highres3dnet_ResExtra.HighRes3DNet_ResExtra *extensions/highres3dnet_ResExtra.py/[CLASS_NAME]*





# Log: 30.12.18
- Working on modifying HR3DN
- What is NonExtraConVal???
  - No Extra Connections Validation: Used to validate the MPRAGE_SPGR_ExtraCon





# Log: 09.01.18
dir > listmyfolder.txt





# MICCAI Meeting Plan
> Possible timeline
> People to get involved?
- Irtiza/ DT developing an intermediate simulator that might be worth looking at: Widen scope of potential sequences to be investigated
> Experiments/Results
- Investigate regression from multiple (similar) protocols to single (significantly different) protocol
  - Should not get very competent results, network cannot differentiate between such similar protocols
- Move onto trying to incorporate sequence paramters into network: If network is more informed regression should perform better
  - Consider even changing activation function: Exponential maybe? (To mimic exponential decays in MR)
> What is the baseline?
- Jog results, intensity standardisations can account for significant inter-sequence differences
> Uncontrollable requirements?
- Data, access, software?
- ???
> Engineering time prediction?
- ???


> Project Plan for forseeable future
Know that mapping from Protocol 1 -> Some protocol 2 should be "easy"
  This is the gold standard

However, when a network tries to train with multiple protocols as input (simulating acquisitions from different sites with slight parameter variations) -> Some protocol 2 it should not perform well
  This is because it is extremely hard to distinguish between images of similar contrasts with differing parameters (Even for a human)

How to circumvent this issue: Feed physics parameters into network paired with corresponding image -> Should provide enough context to network to learn mappings: Know the manifolds of physics parameters

First approach: * Alter HR3DN to add more residual connections
                  * See Highway networks
                  * "All the way" connections, more paired connections
                  * In the ambit of improving identity mapping

**NOTE** Previous work: * Incorrect
                        * Need for each subject to randomly pick a protocol N, not have all N protocols for all subjects



# MICCAI Meeting: Notes
> Teams app/ webapp
Pulsation artefacts MR (Signal related)

> Zach's paper: Style transfer (Scanner property-related things)

> Useful papers
NIPS 2016: Domain separation networks (baseline): https://papers.nips.cc/paper/6254-domain-separation-networks.pdf
Invariant Representations without Adversarial Training: https://arxiv.org/pdf/1805.09458.pdf
Unsupervised Adversarial Invariance: https://arxiv.org/pdf/1809.10083.pdf

> Ze
Curriculum learning in Deep Learning

Jade meeting: Next week, hopefully
Sigma reading group: 3pm Semi-weekly
Weekly group meetings: 11am Tuesdays



Latent space: Structure of data with respect to task at hand

# Domain invariance meeting prep
[Multimodal Unsupervised Image-to-Image Translation]
https://arxiv.org/pdf/1804.04732.pdf
- Goal: Learn conditional distribution of corresponding images in target domain (without seeing examples of image pairs)
- Inherently multimodal
- Existing approaches: Naive 1-to-1 mapping => No diversity in outputs

- Proposal: Multimodal Unsupervised Image-to-Image Translation (MUNIT) framework
- Assumptions: * Image representation can be decomposed into content code that is domain invariant
               * + Style code that captures domain-specific properties

> Style transfer: Recomposing images in style of other images

> If supervised: Conditional generative model OR simple regression model

> Ablation study: Remove some feature from model, see how it performs

[Domain Stylization: A Strong, Simple Baseline for Synthetic to Real Image Domain Adaptation] <2018>
https://arxiv.org/pdf/1807.09384.pdf
> Covariate shift: Difference between training and prediction distributions while output/ concept being learned remains **Stationary**
*Can be a problem when training a network with synthetic images and expecting it to generalise well when applied to real images*

- Goal: Address the problem of covariate shift, wherein problems arise when synthetically trained networks fail to generalise to real data
- Go about this using a modified photorealistic transfer algorithm <This is just a recomposition of an image in the style of another WHILE ensuring that the output image remains photorealistic>
  - Achieve SotA results

- With CG engines have ability to train networks with huge number of labeled training images
- HOWEVER, such networks seem much less accurate
- Some works propose learning a synthetic-to-real image translation function: GAN based. Zero-sum game between discriminator and image translation network. *Goal is to make synthetic image look real*
- Closed form solution
> Contraints enforced to force semantic content preservation

- **[Proposal]**: Domain stylisation using (modified) <Photorealistic> style transfer algorithm: Stylise synthtic images using randomly paired real images

- Clarification: Randomly pair a synthetic image to real image and transferring style (feature transforms)

[Advantages] * No need to train translation network
             * GANs notoriously hard to train: No need
             * "On the fly"

- Notes on other approaches: * CG rendered images only used for DEPTH based object recognition (Not done on RGB yet)
                             *  Semantic preserving loss implemented in GAN related works for S-to-R translation

> Domain adaptation: Special case of visual domain adaptation problem: Adapting a classifier trained in one visual domain to another @MetricLearning @SubspaceModelling @AdversarialTraining

*Domain stylisation step & Semantic Segmentation Learning*
Iterative process: 1. Stylize synthetic image with real image
                   2. Train segmentation network using sylised image
                   3. Use seg. net. to segment real image
                   4. Use segmentation map to better stylise synth. img.
                   5. Repeat.

Baselines: 1. Network trained on just real images (Upper bound)
           2. Network trained on just synthetic images
           3. DOmain randomisation: Introduce variability into synthetic images (texture on object meshes, hue shift)
           4. CycleGAN: Image-to-Image translation model

Results: * Performs exceedingly well, particularly in indoor settings. Followed closesly by CycleGAN
           * Distances between synthetic and real images drastically reduced

[Domain Separation Networks]
> Similar to previous paper: * Gathering a lot of real data is expensive
                             * Large cost of large data collection
                             * Why not use synthetic images?
                               * Failure to generalise: Need domain adaptation algorithms to come into play

- Existing methods: 1. Mapping representations from one domain to other
                    2. Learning to extract features that are invariant to domain from which they were extracted

- Proposal: * Explicitly model what is unique to each domain
              * Should help model's ability to extract domain-invariant features
              * Outperforms SotA in UNSUPERVISED settings

Baselines: * Adversarial training to find domain invariant representations (DANNs)
           * Maximisation of the domain classification loss replaced by minimisation of Maximum Mean Discrepancy (MMD) metric

<Method>
Image classification as as cross-domain task
* Labeled dataset in source domain
* Unlabeled dataset in target domain
  * Want to train classifier on data from source to generalise to target
  * Want to capture representations shared/ not-shared by domains
    *Obtain more meaningful representations*
> Autoencoder: Unsupervised learning algorithm that applies backpropagation, sets targets == inputs

[Unsupervised domain adaptation in brain lesion segmentation with adversarial networks]
Two-fold training: * Segmentation loss paired with domain accuracy loss
                   * Want to minimise segmentation loss while maximising domain accuracy loss
                     * Ensures domain adaptation since it pushes network to learn mappings invariant to differences between domains

[Multimodal Unsupervised Image-to-Image Translation (MUNIT)]
- Goal: Learn conditional distribution of corresponding images in the target domain (unsupervised)

- **[Theory]**: * Assume that image representation can be decomposed into <Domain-Invariant> **CONTENT CODE** + **STYLE CODE** that captures <Domain-specific> properties
                * Translation: Recombine CONTENT CODE with <RANDOM> STYLE CODE sampled from style space of target domain

- Introduction: * Task of domain translation far more challenging when unsupervised
                * CrossDomain mapping (of interest) typically multimodal
                  * Existing techniques assume deterministic/ unimodal mapping

- Methodology: * Images in each domain are encoded to:
                 *Shared Content space*
                 *Domain specific style space*
               * For translation: 1. Recombine content code of input
                                  2. With random style code of content space
                                 (3. Different code -> Different image)

- Assumptions: * Image: x_i { X_i
               * Have marginal distributions: p(x_i)
                 *No access to joint distributions*
               * GOAL: + Estimate conditionals p(x_1 | x_2)
                       + Complex, =/= deterministic
               * Partially shared latent space assumption
                 *Image formed by shared + unique space*
               * x_i = G_i(c, s_i) *G == generator, E == Encoder*

- Model: <Translation>: 1. Extract content code from input (Encoder)
                          2. Randomly draw style latent code from prior dist. *Element of Natural numbers*
                          3. Use G2 to produce final output

         <Loss>: * Bidirectional reconstruction loss *G == Inv(E) VV*
                 * Adversarial loss *Match trans. img. to target dom.*
                   * GANs

-



Autoencoder: * <Learning> efficient data coding (unsupervised)
             * <Learn> "representation" for a set of data, for dimensionality reduction
             * Then, <Reconstruct> (generative)
             * <By engaging in "dimensionality reduction" can, for example ignore noise>

*Tie in with <LATENT-VARIABLES>: * Variables lying in the space of the bottleneck layer





# Log:
Aims:
- Alter HR3DN to allow for concatenation of physics parameters
  *Only TI initially*

[Physics parameter concatenation] ##APPLICATION
- Change regression_task.py: 1. Created new application under </home/pborges/NiftyNet/niftynet/application> named @regression_task_fcFeature
                             2. Keep class name the same for now (RegressionApplication)
- Change highres3dnet: 1. Created new version under </home/pborges/NiftyNet/extensions> named @highres3dnet_ParamConcat
                       2. Added fc_feature as new variable
- Change .ini:
  *See Pritesh's files for starting point to modifications*
- Further notes: 1. In qsub files MUST call with **net_run.py -a **niftynet.application.[regression_application_fcFeature].[RegressionApplication]** to ensure that NiftyNet looks at desired application (Calling net_regress.py will just fetch the vanilla regression application)





# Log:
Aims:
- Continuation of work to alter HR3DN

[Layer structure]
- HR3DN has no FCL: * To concatenate parameters need to add such a layer
                    * <Altered> @highres3dnet

**[MUNIT PAPER: SALIENT POINTS]**
- Disciminator (i.e. Adversarial GAN loss) forces the similarity in content codes because it pushes the generators to produce images from domain 1 that are similar (when translated) to domain 2: Only going to happen if there exists some shared features

- L1 loss between randomly drawn s1 and extracted s1_hat enforces the requirement that s1_hat follow a normal distribution

- Losses: Image reconstruction loss <Sharpness>
          Latent reconstruction loss (C + S):
            * <ContentCode> Encourages preservation of semantic content
            * <StyleCode> Encouraging diverse outputs





# Log:
- Build body of literature relating to: 1. Regression and CNN architectures
                                        2. Domain adaptation/ transfer networks
- Notes on FC HighResNet
- DSE to (say) MPRAGE? Multi-fold M-MPRAGE -> SPGR?

[HighResNet]
- Tried adding FC layer to end of HighResNet: **ERRORS**
  - Cannot return 3D "image" from a FCL
  - Problems relating to <CropLayer>: Expects 3D layer, FCL is 2D
    *Similar issues relating to ResNet: Final layer is FC*

[Invariant Representations without Adversarial Training] ##InvAdvTrain
> Fair encodings: Remove certain features so as not to bias task towards such features (e.g.: Sex, race)

~ KL divergence





# Log: 21.01.19
Aims: - Continue with desktop setup _WIP_
      - Ivana reminder email (again...) [Y]
      - Continue reading of ##InvAdvTrain
      - Microsoft Teams: Update plan





# Log: 22.01.19 ##GIT
Log: - Write down git repo management guide
     - Config guide

[GIT: Pushing local repo to GitHub] 1. git remote add <NAME> git@github.com:<USER>/<REPO>.git
                                    2. Generate ssh key: ssh-keygen -t rsa -b 4096 -C "<EMAIL>"
                                    3. Check ssh-agent is running: eval $(ssh-agent -s)
                                    4. Add key to SSH agent: ssh-add ~/.ssh/id_rsa
                                    5. Add key to GitHub account: Settings -> SSH and GPG keys -> New SSH key
                                    6. git push -u <NAME> <RepoBranch>

[GIT: Comparing remote to local] 1. git fetch <REMOTE> <LOCAL>
                                 2. git diff <REMOTE> <LOCAL>

[Host config] 1. **Locally** Generate SSH key ##KEYS
                             Copy key to clipboard

              2. **TUNNEL** Add <LOCAL> key to authorized_keys
                            Generate SSH key
                            Copy key to clipboard

              3. **Cluster** Add <TUNNEL> and <LOCAL> key to authorized_keys

[Config Host]
Host comic2
       User <CS_USERNAME>
       HostName comic2.cs.ucl.ac.uk
       proxyCommand ssh -W comic2.cs.ucl.ac.uk:22 <CS_USERNAME>@storm.cs.ucl.ac.uk

[Drive mounting] 1. sudo apt install sshfs
                 2. sshfs comic2:<DIR> <LOCALDIR>

[Repeat tasks] crontab -e: Edit file according to needs *Probably worth it for rsync*
                           <minute> <hour> <DayOfMonth> <Month> <DayOfWeek> [COMMAND]
                     e.g.:    0        5        *          *         1    sudo apt-get update (Update repos. once a week at 5am)

**[RSYNC]**
@Resources: https://www.tecmint.com/rsync-local-remote-file-synchronization-commands/ (Technical)
https://medium.com/@sethgoldin/a-gentle-introduction-to-rsync-a-free-powerful-tool-for-media-ingest-86761ca29c34 (Large overview)
- Allows for copying files to/ from local/ cluster
- Typically quicker than scp
- Can just call directly now that have set up Host Configs locally
- Recursive + Incremental method

> Syntax: [rsync] -a -i -h --progress <FILE> <TARGET>
*Can/ should use in combination with crontab for continuous updating*

> Other Notes: * Install zsh + Oh My Zsh
               * Look into <vimtutor>

[Terminal] ##Terminal
- Throw process into background: <CTRL-Z> + <bg>
- Bring process into foreground: <fg %ID>  *Just fg if there's only one paused job*
- Kill a background process: Find process ID <jobs>
                             <kill %ID>





# Log: 23.01.19
Aims: - Begin work on Experiment One _WIP_
        - Generate images for ONE subject
        - Choose one sequence (MPRAGE?), reasonable parameter variation
        - Regress parameters (TR, TE, TI? FA?)
      - Install zsh, OMZ [Y]
        - SE
      - Run test with NiftyNet on Desktop [Y]
        - @230119_Res

[NiftyNet Test]
> Packages to install: tensorflow-gpu==1.10, numpy, scipy, blinker, pandas *Using pip3*

**NOTE** Do <NOT> upgrade pip3 (something to do with it being the system version)
         See: https://stackoverflow.com/questions/49836676/error-after-upgrading-pip-cannot-import-name-main

         Do <NOT> install latest version of numpy (Tables library issue?) *Go for numpy==1.14.5 instead, less than 1.16 for sure*
         See: https://stackoverflow.com/questions/54200850/attributeerror-tuple-object-has-no-attribute-type-upon-importing-tensorflow
              https://stackoverflow.com/questions/54196106/error-with-calling-numpy-scipy-gensim-in-python3/

         ZSH shell is <NOT> bash! Modify ~/.zshrc INSTEAD OF ~/.bashrc *Had problems with tensorflow recognition since rc had not been sourced*

         Do <NOT> need to use submission scripts, can just call w/ python(3) directly *Make sure to take care with directories, use absolute*

[Results] ##230119_Res
- First attempt successful, copied <4MPRAGE_SPGR> folder from cluster and ran training w/ 2500 iterations
  - Considerably slower than cluster, ~2.4s per iter, vs ~0.5 for cluster
  - Good for debugging, registration, "quick" tasks

# Jorge Meeting
- Main issues: * Hard for humans to account for physics parameters when segmenting
               * Hard in general for bias to be accounted for: Can "acknowledge" it to little effect

- Next steps: * Regress physics parameters from SINGLE patient (Consider dimensionality, few hundred images)
              * Perform segmentations (Use GIF NOT CNN) given multiple realisations of same phenotype (i.e. same sequence, different parameters)
                *Find appropriate network to be able to inject physics knowledge and therefore reduce uncertainty on volumetric predictions*

[Experiment One: Parameter Regression from single subject]
- Two main sub-experiments: 3D + 2D
- (Initial) modality of choice: MPRAGE (High spatial resolution in real world setting ~ MPMs)

> Todo: Find realistic range of MPRAGE parameters
>       Pick "healthy" subject: Subject 04





# Log: 24.01.19
Aims: - Experiment One: Continuation _WIP_
        - Generate images
        - Move to Desktop

[Experiment One: Parameter Regression]
# MEET TOM 1:30 FRIDAY 25TH
Parameter Regression: Will require modification to application + new code

Look into: - Application driver

[Application driver]
When calling net_[APPLICATION], imports from NN/nn/__init__ == "main"
In turn, this imports from NN/nn/engine/ApplicationDriver (Something like that)
Graph is created here according to application type





# Log: 25.01.19
Aims: - Begin work on GIF segmentations
      - Begin implementation of parameter regression funcionality using NiftyNet

[FSL Installation] ##FSL_INSTALL
- Download: fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux
- Run installer with python (2)
- Add FSLDIR to **.profile**
  - *Also apt install libopenblas-base*

[GIF Segmentations]
- Need to figure out how to carry out segmentations using GIF

# Tom meeting: Parameter regression
- Cloned Tom's NiftyNet fork: https://github.com/tomvars/NiftyNet.git */home/pedro/NiftyNet-T*
  - Opened in pycharm as a project, important REF files: **modality_classification_application.py** *Information relating to application*
                                                         **train_modality_classification_Lesions_MICCAI.ini** *Information about configuration*





# Log: 28.01.19
Aims: - Continue work on implementation of parameter regression functionality

[Parameter Regression]
- Created modified regression application: </home/pedro/NiftyNet-T/NiftyNet/niftynet/contrib/harmonisation/param_regression_application.py>
  - Have to modify, accordingly, **user_parameters_custom.py** in </home/pedro/NiftyNet-T/NiftyNet/niftynet/utilities> ##CUSTOMREG ##CUSTOM_REG
    - Specifically, create new function (named __add_regression_args_Param) to allow for new parameters (Specifically, 'parameter') in config

**NOTE**: - Naming convention: Subject IDs lose "dots" in filename, have to name csv file accordingly
          - Regression applications: **num_classes = 1** *See default regression_application.py*
          - Currently getting error associated with loss: Shape is zero, apparently @TBC
            - Something related to extracting "parameter" from the csv file





# Log: 29.01.19
Aims: - Complete slide for AMIGO meeting
      - Continue work on parameter regression application (2D)
      - Begin work on GIF segmentations

[AMIGO Meeting]
...

[MPRAGE Optimisation]
> Optimizing the Magnetization-Prepared Rapid Gradient-Echo (MP-RAGE) Sequence
- Good paper discussing optimisation of MPRAGE protocol (according to TI) *Also mentions decay time (TD) and echo spacing (tau)*
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4039442/
  - <TI: 880 - 1200 ms
  - <TR: 1950 - 2530 ms
  - <tau: ~ 10 ms
  - <TD: Conflicting, optimal~ 600 ms (In 'literature'), but this paper suggests low == better *for SNR, NOT CNR*

[Parameter regression]
- Debugging: 'parameter' variable cannot be found by NiftyNet for some reason
  - Rectified configuration file: Should not have any header (As opposed to subject_id,label as specified in csvReader)
    *Tom will look into this tomorrow* @30th

[GIF Segmentations] ##SEG_EM  ##GIF_INIT
- Just use seg_EM for now:                          [seg_EM] -in <FILE> -out <OUT> -nopriors <NUMCLASSES>
  - Preliminary results found in </home/pedro/NiftyNet-T/data/Segs> *TI ranges: 850 - 1000 ms*





# Log: 30.01.19
Aims: - Parameter regression implementation + literature review
      - Segmentations

[Parameter regression]: **WORKING**
- Changes to get it to work: * Found that variables were getting instantiated that were unwanted
                              * Only need parameter + image (and respective locations)
                              * Instead, was also getting error maps, loss borders, etc
                              * **BECAUSE** uniform_sampler was creating dictionary from <TaskParams> instead of <CSVReaderName> *sampler_uniform_v2_csv*
                             * Also found that application is classification "oriented", treats parameters as individual classes instead of as continuous variable
                               * **SOLVED** create to_continuous method of CSV reader (instead of using to_categorical) *csv_reader.py in contrib*
                                 *What is the difference though?*

Working directory: </home/pedro/Project/ParamRegression/TomTests>
- Tasks: * Ran on 250 images (80/10/10) [Y]
          * Obvious overfitting, see model folder
         * Set to run on 5000 images (TI: 800-1200) overnight _WIP_

[Other useful notes] * kill -9 <TASKID> to kill a task (e.g.: something running on GPUs)
                     * sleep <N>; [COMMAND] to run a command after <N> seconds

[GIF Segmentations]
- Running GIF segmentations for 50 MPRAGE images (Same patient) for TI in range(800, 1200)
  - GIF found on cluster under </share/apps/cmic/GIF>, running the [/share/apps/cmic/GIF/runGIF_v3_folder.sh @FOLDER] script
  - Queues everything on cluster automatically





# Log: 31.01.19
Aims: - 5000 image parameter regression
      - GIF Segmentation check
      - JES **Due 7 Feb**

[5000 image parameter regression] - **ERROR** Did not save images with names with enough decimal places -> Name Overlap
                                  - Re-ran image + csv creation process, with TI range 800 - 1300 (5001 images for 0.1ms gap)

[GIF Segmentations] - Awaiting results, all jobs running properly AFAIK


Basal: 37, 38, 58, 59, 24, 31, 56, 57
       37, 38, 58, 59, 24, 31, 56, 57, 61, 62
       basal_labels = np.array([37, 38, 58, 59, 24, 31, 56, 57, 62, 63])
       # 62, 63 are the ventral DC (https://mail.nmr.mgh.harvard.edu/pipermail/freesurfer/2013-May/030388.html)





# Log: 06.02.19



# Jorge Meeting
Salient points: - Problem well motivated for 2D (such that no need to worry about 3D just yet)
                - Seems like parameter regression works decently well for single subject: Try adding in dementia subject

**NETWORK**: SEGMENTATION TASK
- _Simple_ approach: * Use **MPM Segmentation** as SOTA GT (e.g.: T1 map for MPRAGE GT)
                     * Pass physics parameters into small, subsidiary network (e.g.: Two FCL)
                       * Append parameters as extra channels to output of network of choice (e.g.: HRN)
                       * Ensure that gradient back. prop. also updates subsidiary network
                **NOTE** [Assumes that anatomical representation (network output) is "complete" and only needs additional physics knowledge]
                     * <Alternative>: Pass parameters earlier on in the network
                **NOTE** [Assumes that network needs physics parameters early on because they encode useful information for spatial learning]





# Log Catch-up
Finished GIF segmentations (MPRAGE): Plotted region volume as function of TI <Project/ImageSegmentation>
Parameter regression experiments

Carried out segmentations of smaller regions (DGM, Cortex) (Wed/ Thurs)
Preliminary CNR/ SNR analysis carried out (Wed/Thurs)
Continued parameter regression tests on 20k slices (Wed/ Thurs)

Created split MPM files (T1 maps) for ground truth segmentations (Friday) </home/pborges/AD_MPM/MPMs/Split/T1s>
Set up GIF segmentations for T1 maps (Friday)





# Log: 11.02.19
Aims: - Carry out parameter regression with extended TI (0.2 min instead of 0.3)
        - Will give greater insight into nature of datasplit bug
      - Compute and plot tissue and sub-tissue volumes: See <Project/ImageSegmentation/SNR_CNR/Scripts/main_tissue_volumes.py>
        - Of particular interest: Segmentations of extended TI range volumes <GIF_Many_Params> **Home directory full**
      - Investigate T1 map (GT) segmentations **Home directory full**

[Segmentations]
**NOTE** Need to sort directories beforehand (See sort_dir function)
         plt.show() after plots to actually see plots

[Parameter Regressions]
Might as well leave this until tomorrow/ Wednesday when Titan V is available

Experiments: - Noticed that when training with JUST training (no val or inf), behaviour is VERY different depending on csv shuffling
               - No shuffling: Great
               - Shuffling: No better than guessing the mean
                 *Investigate further tomorrow* <N>





# Log: 12.02.19
Upgrade Presentations all day event





# Log: 13.02.19
Solving the import error:
> Right-click -> Mark Directory as -> Sources Root
Adding new loss function: 1. Create function in relevant loss file (e.g: loss_regression.py)
                          2. Edit factory application to add supported alias <niftynet/engine/application_factory.py> ##FACTORY ##NNALIAS
                            3. e.g.: "L3Loss": 'niftynet.layer.loss_regression.l3_loss'

Tested out inference on 18k dataset for different loss functions: - L2Loss: Decent
                                                                  - L3Loss: Best
                                                                  - RMSE: Decent, odd results for low TI
> Used .../Simulations/nii_param_extractor.py for this

# Jorge catch-up meeting
Discussion of inference curves: Greatest divergence between prediction and real on extremes of dataset
  Makes sense: * All network tries to do is minimise the loss function given to it.
               * With L2 loss there's no reason why the network would "risk" high predictions (for large TIs)
               * Overall the fit minimises the L2 (Over-estimation on small TI, overestimation on high TI)





# Log: 14.02.19
- Started work on implementing physics parameter injected version of HR3DN
  - Considered using Unet, but HR3DN has 2D/3D flexibility + more familiar with code

[General Notes]
_Printing_: print(<TENSOR>.get_shape().as_list())
_param_regression_application_: ResNet is **HARD-CODED**, version found in </niftynet/contrib/pimms/resnet_plugin.py>





# Log: 15.02.19
Aims: - Keep working on physics injection in HR3DN <niftynet/contrib/harmonisation/highres3dnet.py>





# Log: 18.02.19
Aims: - Keep working on physics injection in HR3DN: 1. Create Physics branch
                                                    2. Successfully merge branch with main network body
                                                    3. Implement means of feeding in physics parameters into network (via config + CSV?)
      - Test out network implmentation with regression first?

*[Asides]*
- More robust directory setup script for NiftyNet jobs
- Create environment for testing out tensorflow (instead of having to rely on colabs)

[Pycharm] ##PYCHARM
          - Ctrl + Alt + e  *Command history (ease of copying/ searching)*
          - Ctrl + Alt + Shift + j *Mass replace selection in editor*
          - Ctrl + Shift + f *Find something on project level*

[GIF Segmentation MPM]
- Bad segmentations of T1 volumes: - Due to bad registration during aladin portion of segmentation
                                     - Likely because of extra-cranial hyperintensities in T1 volume
                                   - Solutions: 1. Try to segment R1s
                                                2. Dilate TIV significantly + Gaussian smooth and use as rough mask before segmentation



# Log: 19.02.19

[RegressionTest]
- Problems with regression application of NiftyNet-T: - Unknown input error
                                                      - BUT regression application files are identical
                                                        - Factory settings?

**SOLVED** - Problem associated with modality dropout in <NiftyNet-T>
             - Moved uniform_sampler_v2 from <NiftyNetLocal/.../engine> to <NiftyNet-T/.../engine>
             *Aside*: Added summary for regressed image in <NiftyNet-T/.../regression_application.py> (i.e.: net_out)
                      Can see image evolve in tensorboard


# Log 20.02.19  ##Segmentations
TIV Mask
- Use seg_maths to dilate first iteration GIF TIV by 10 pixels: [seg_maths <FILE> -bin -dil <PIXELS> <OUTPUT>]
  - Use dilated TIV as mask for OG R1 image: [seg_maths <FILE> -masknan <DIL_TIV> <OUTPUT>]

- Wrote small bash script to automate above process: </bin/mass_dil_masker.sh> <WorkingDirectory> <DilationPixelNumber>
  - Currently running under </GIF/.../Masked_OGs>

[Setting up test segmentation experiment] ##R1s
- Create labels: Argmax across single (middle) slice of GIF segmentations (found under </GIF/.../R1s/Segmentations) <181>
- Network requirements: 1. Input images *MPRAGE slices* [Y]
                        2. Segmentation labels *GIF segmentation slices* <N>
                        3. Physics parameters *Inversion time nifties* (Extracted using nibabel + [0][0][0])  [Y]


*Quick coding to extract relevant segmentations from full GIF segmentation* </home/.../Simulations/Argmaxer.py> ##Argmaxer
Skull is kept so that the CSF label does not become zero, but the skull does (which we don't want anyway so can be set to zero no problem)

# Read in the file and get the first 4 volumes (Skull, CSF, GM, WM, DGM)
>>> sub_21, aff = read_file('R1_split_sub_21_0000_NeuroMorph_Segmentation.nii.gz')
>>> sub_21 = sub_21[:, :, :, 0:5]

# Sum the GM probabilistic segmentations to arrive at total GM probabilistic segmentation
>>> tot_GM = sub_21[:, :, :, 2] + sub_21[:, :, :, 4]

# Remove the now useless DGM segmentation
>>> sub_21 = sub_21[:, :, :, :4]

# Set the "GM" part of the 4D seg volume to the total probabilistic GM volume
>>> sub_21[:, :, :, 2] = tot_GM

# Argmax across the 4th dimension and save image
>>> argmax_sub_21 = np.argmax(sub_21, axis=3)
>>> save_img(argmax_sub_21, aff, 'argmax_sub_21_prop.nii.gz')


**NOTE**
Could have used seg_maths with -tpmax flag BUT no easy way to remove unwanted seg. classes + combining GM w/ DGM



[SFORM]
fslorient -setsformcode 2 MPRAGE_TI_0.92820.nii.gz
fslorient -setsform -1 0 0 89 0 1 0 -125 0 0 0.5 19.5 0 0 0 1 MPRAGE_TI_0.92820.nii.gz

File "/home/pedro/NiftyNet-T/NiftyNet/niftynet/contrib/harmonisation/user_parameters_parser_tom.py", line 192, in run
  input_data_args[section].csv_path_file,
AttributeError: 'Namespace' object has no attribute 'csv_path_file'





# Log: Friday
Many points to make!
- Physics param printing: Printing whole array NOT [BS x 1 x 1 x 1]
- Successfully implemented Physics params by passing Image sized volume
  - Max (tf.reduce_max()) across dimensions 1 + 2 (leave BS dimension alone)
- TF logging: Output_prob False for inference: Label map therefore
              To get label map during training have to argmax net_out before passing to TF_SUMMARIES *Also have to convert to uint8 beforehand*
                Extra note: summary type is important, use [image] for 2D slices
  - 3 images being displayed: Get around by passing a single example instead of whole batch (i.e.: index net_out accordingly)





# Log: Sunday
- Aims: Run simulations for SPGR (slices)
          Go for multiple subjects (Healthy 21 and Unhealthy 5??)
        Fix HRN-P: Physics concatenation not flawless, adding too many channels

Linspace: Increments = spacing / (num_samples - 1)

[Debugging]
Printing out final channel in flow_concat in HRN-P: * Should be repeats of the [physics_param]
                                                    **WORKS** Console printing showed this to be true *Need to test on SPGR*

[SPGR Tests] </home/.../Project/SPGR_Test>
Generated dataset of 3775 slices of params: <TR> 20 - 80 ms (151 samples)
                                            <TE> 4 - 16 ms (25 samples)
                                            <FA> 15 ms (constant: 1 sample)


file:///
Found under: </home/pedro/Project/GIF/AD_MPM/Segmentation_slices/GroundTruth_subject_21_2D_Segs_3775_SPGR> [Labels]
             </home/pedro/Project/Simulations/Single_Subject_21_SPGR_3775_Params> [Slices]
             </home/pedro/Project/Simulations/Single_Subject_21_SPGR_3775_Params_TI_slice> [3D Param slices of TR + TE]

**NOTE** Need to figure out how to differentiate broadcast_to!!! <HP>
  Alternatives: - Tile
                - Stack
                - Ones x Values + concat





# Log: 25.02.19
Aims: - Fix broadcasting issue (See previous log for ideas on potential solutions)
      - Fix image sampler issue: i.e.: Problems because param nii size =/= image size

[Ones attempt] * Create tf.ones() array with shape == [BS, Image_Y, Image_X, N_feat] == flow.shape[:-1] + [N_feat] <A>
               * Expand dims. of physics_flow: [BS, N_feat] --> [BS, 1, 1, N_feat] <B>
               * physics_flow_concat = <A> x <B>
**NOTE** This worked!

[Fixing Sampler Issue]
Problem Summary: * NiftyNet finds shapes incompatible for sampling purposes
                  * Specifically Image size (e.g.: 200 x 200) vs Param size (1 x 1 x 1)
                  * It tries to sample them accordingly, but this does not work, obviously

Potential solutions: - Pass <Param> as separate entity, not subject to sampling *Segmentation Application changes required* <App>
                     - Make/ delete exception in sampler to circumvent error *Uniform sampler changes required* <Samp>
                     - Save parameter arrays as randomly sampled from range of parameters

> Samp
Problem arises in the rand_spatial_coordinates function: Compares shapes and raises error if not the same


**<POSTPONE>** Seems like this is a bigger task than it looks, work with just MPRAGE + TI for now
               *FIXED* See 26.02.19







# Log: 26.02.19
- Exponentiation as additional parameters to pass to network
- Motivation: Well motivated as per tissue volume vs TI plots
  - **DO** Investigate with unhealthy subjects (10 to start)

# Sampler changing adventure
Data passed to (Uniform) sampler as dictionary of name, array pairs *'image', 'physics_params', 'label'*
From this, [image_shapes] is created which is a dictionary of name, array SHAPE pairs
  - Problem arises here: This is passed to a [match_shapes] method which compares array shapes with window shapes (from config)
  - Physics params shape > label/ image shapes (Since parameters encoded in the 3rd dimensions) -> <FAILURE>

**SOLUTION**: Create two separate dictionaries for [DATA] and [IMAGE_SHAPES]
                One for Image + Label, One for physics_params
                Shape matching doesn't return an error since physics_params is passed and only compared against itself

[Masked R1s] **[mass_dil.sh <directory> <dilation>]**
- Reminder: * Segmentation likely to perform better if mask the image according to TIV
            * Had tried doing so with dilation = 10 </comic/cluster/project0/possumBrainSims/GIF/AD_MPM/MPMs/Split/T1s/R1s/smol_masks>
              * **ERROR** Obtained segmentations looked less than ideal, excessive skull cropping
              * <Retrying> with dilation = 25 <GIF/.../Dil_masks_v3>





**[IMPORTANT DIRECTORIES]** ##VIPDIR ##DIRECTORIES
- Single subject 21 segmentations (High TD) </home/pedro/Project/ImageSegmentation/FilesOnly/Segmentations>
  _PERCEIVED segmentations, not be to confused with "Ground Truth" segmentations derived from R1 maps_

- Single subject 21 segmentations (Low TD) </home/pedro/comic/cluster/project0/possumBrainSims/GIF/Single_Subject_Many_Params>

- Single subject 21 MPRAGE realisations (Low TD) </home/pedro/Project/Simulations/Single_Subject_Many_Params>
  _Can use these with MPRAGE_2D_Single as IMAGE_

- Single subject 21 MPRAGE realisations (High TD) </home/pedro/Project/Simulations/Single_Subject_Many_Params_High_TD>
  _Can use these with MPRAGE_2D_Single as IMAGE_

- Single subject 21 TI slices (2D) </home/pedro/Project/Simulations/All_subjects_TI_slice_2D_[train/inf]>
 _Can use with any experiment involving passing TI as parameter to the network_

- All subjects GT segmentations (R1 based) </home/pedro/Project/GIF/AD_MPM/R1_segmentations/Labelled_segmentations>
 _Use as Labels in network training since want to have network learn invariant representation_

- Inference subjects GIF segmentations </home/pedro/Project/GIF/AD_MPM/Segmentation_volumes>
- Inference subjects Baseline segmentations </home/pedro/Project/ImageRegression/Baseline_All_inf>
- Inference subjects Physics segmentations </home/pedro/Project/ImageRegression/All_subjects_physics_inf>
 _Various results from the respective trained networks (Or non-DL software GIF)_





# Log: 27.02.19
Aims: Begin preliminary training on Physics informed network (Single param)
      Ideas: Pass also exponentiation of parameter

[Preliminary training] <Project/ImageRegression/MPRAGE_2D_Single>
- Issue initially: Mismatched names: Turns out 3D data is following old naming convention (4f instead of 5f)
                                     - Parameter files are new, therefore were following newer convention, hence the mismatch
                                       - Fixed: Use 4f instead of 5f

Images: </home/pedro/Project/Simulations/3D_Single_Subject_21_MPRAGE_50>
Labels: </home/pedro/Project/GIF/AD_MPM/Segmentation_volumes/sub_21_proper>
Parameters: </home/pedro/Project/Simulations/3D_Single_Subject_21_MPRAGE_50_TI_slice>





# Log: 28.02.19
Aims: Investigate if network is training properly

[Investigating training]
Want to know if physics branch is training properly: Visualise weights in Tensorboard
Insert the following line in <engine/application_variables.py> under _finalise_output_op_ function:

file:///_Distribution_&_Histograms
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for var in all_vars:
   tf.summary.histogram(var.name.replace(':', '_'), var)

**ERROR** Weights do not evolve at all for FCLs: * Try freezing other gradients
                                                 * Reformulate way in which physics params are transferred to main network
                                                 * Start over with fresh version of Niftynet


> {HRN gradient freezing}
- Insert following line in physics_segmentation_application after 'grads' definition:
                                grads = [(grad, var) for grad, var in grads if 'FC' in var.name]
- Ensures that ONLY those gradients associated with the FCLs are updated

*OUTCOME*
<NONE> Did not work, weights still do not update



> {Physics transfer reformulation}
- Instead of creating an array of ones and multiplying output of Physics branch by it, use <TILE>

flow_concat = tf.concat([flow,                                                        *Concatenation with flow*
                         tf.tile(tf.expand_dims(tf.expand_dims(physics_flow, 1), 1),  *Need to expand middle dimensions for tiling*
                                 [1, flow.shape[1], flow.shape[2], 1])], 3)           *Tile along middle dimensions to match dimensionality of flow*

*OUTCOME*important direc
<NONE> Despite far cleaner appearance, did not work, weights cease to update



> {Starting over from fresh version of NiftyNet}
- Move away from forked Tom NN and clone newest version (0.50) of NiftyNet and transfer work there
- Cloned into: <NiftyNet-5>

*OUTCOME*
[SUCCESS] Weights change as network trains (albeit only a little)





# Log 01.02.19 + 05.03.19
Aims: - Set up training of ALL SUBJECTS
        - Simulate MPRAGE volumes for all subjects: 700 - 1200 ms (as well as TI slices)
        - Use GIF segmentations of R1 as labels

[Physics]
> Training/ Inference split
Inference: Subjects {2, 6, 11, 14, 25} *Subject 5 excluded completely due to failure of segmentation*

Images: </Project/Simulations/All_subjects> + </Project/Simulations/All_subjects_inf>
Parameters: </Project/Simulations/All_subjects_TI_slice_train> + </Project/Simulations/All_subjects_TI_slice_inf>
Labels: </Project/GIF/AD_MPM/Segmentation_volumes/All_subjects>

[Baseline] (Cluster)
> Resource requests
tmem=11.5G, h_rt=99:59:0

Images: </cluster/project0/possumBrainSims/SegTests/Data/All_subjects>
Labels: </cluster/project0/possumBrainSims/SegTests/Data/All_subjects_labels>

[Setting environments up appropriately]
Massive pain to set up labels, process was as such:
<maker.sh> Creates appropriate subdirectory structure for images under </home/pedro/Project/Simulations/All_subjects_label_templates/>
           Need to separate volumes to allow for use of this as a template for <same_file_many_names.sh> [TEMPLATE_FOLDERS]
<copier.sh> Copies each subject's volumes into the appropriate [TEMPLATE_FOLDERS]
<label_copier.sh> For each subject, calls <same_file_many_names.sh> with [TEMPLATE_FOLDERS] as templates
                  Creates labels in </Project/GIF/AD_MPM/Segmentation_volumes>
                  Copy TRAINING subjects into single folder, <All_subjects>


[Additional notes]
- Could be beneficial to have square inputs (i.e. dimX == dimY) to not have kernel bias
- Could also be beneficial to just crop/ pad image to required size by the network (i.e. 8 divisible)
- For padding: NiftyNet *Volume_padding under [NETWORK] section*
- For cropping: fslroi *Example: fslroi <INPUT> <OUTPUT> 3 176 1 216 0 -1 0 -1*





# Log: 06.03.19
Aims: - Investigate status of [Baseline_All] & [All_subjects_physics]
      - Set up SPGR experiments: * Simulate volumes _WIP_
                                 * Labels already present
                                 * Different split? (Start with same for now)

**[SLICE MISTAKES]**
TI_slices are 4D! Accidentally made it size of MPM instead of just 2D
  As a result (because only 3 dimensions are specified in spatial_window_size in CONFIG) sampler takes ALL of the 4th dimension (Oops)
  Should be fine for now, but change for future experiments

[SPGR]
Simulated with: * Varying TR </Project/Simulations/SPGR/Subject_21_TR>
                * Varying FA </Project/Simulations/SPGR/Subject_21_FA>

Differences seemed minor, currently running segmentations using GIF on both folders </cluster/project0/possumBrainSims/GIF/Subject_21_TR>
                                                                                    </cluster/project0/possumBrainSims/GIF/Subject_21_FA>

Also segmenting ALL inference subject volumes (MPRAGE): </cluster/project0/possumBrainSims/SegTests/Data/GIF_All_Subjects_inf>





# Log: 07.03.19
Aims: - Investigate status of [Baseline_All] (~2.8k) & [All_subjects_physics] (~14k) *Consider evaluation at 5k-ish*
      - Begin writing _WIP_

[Writing]
...

[Status]
- Progressing just fine, might leave until 10k iterations potentially
- Carry out to 15k





# Log: 08.03.19 + 12.03.19
Aims: - Investigate status of GIF segmentations for all subjects
      - Investigate volume differences for Physics vs Non-Physics vs GIF
      - Keep writing in preparation for tomorrow _WIP_
        - Progressing nicely, [Intro, Methodology, Experiments]

[Inference GIF segmentations]
- All seem to have completed: - 8 files missing (298 files instead of expected 306)
                              - Currently copying files into <home/.../Project/GIF/AD_MPM/Segmentation_volumes>
                                - *Should be able to identify missing files after the fact*
                              - Re-running missing segmentations: </cluster/project0/possumBrainSims/SegTests/Data/GIF_All_Subjects_inf/redos_{}>
                                - Subject 2: 1.10, 1.11, 1.13
                                - Subject 11: 0.87, 0.88, 1.12
                                - Subject 14: 0.83, 1.03

[Extras]
- Finally running segmentation of subject 5 R1 map </cluster/project0/possumBrainSims/GIF/AD_MPM/MPMs/Split/T1s/R1s/OGs/5> (Unmasked)





# Log: 13.03.19
- MICCAI paper draft meeting
- Look into MR acquisition physics
- Plot GIF segmentations against Base + Physics


[MR Resolution Notes] Resolution: Δx = 1/(NΔk) = L/N_image (L = 1/Δk is the FOV)
                                   Δk_R = γGΔt (Δt == spacing between readouts in time)

Need to run argmaxer on GIF segmentations for extra baseline validation
SPGR experiments: Randomly sample from space of variables for training ##SPGR_RANDOM

Moved Finalised GIF segmentations of inference subjects into appropriate local directory </home/pedro/Project/GIF/AD_MPM/Segmentation_volumes/Inf_Labels>
Altered <inference_validation.py> to show GIF segmentations

[SABRE DATA: 20 subjects] </home/pedro/Project/Data/SABRE>
Acquired data from Carole: * Three T1 acquisitions for same subjects
                             * Two MPRAGE, One TSE

**NOTE** No inversion time mentioned in .json files for either MPRAGE sequences
         5mm slice thickness for TSE acquisitions

[TSE] http://mriquestions.com/what-is-fsetse.html
Works by reading multiple k-space phase-encoding lines per TR interval

[MPRAGE]
- Important parameter discussion paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4039442/pdf/pone.0096899.pdf (ADNI vs Siemens vs Freesurfer)

[MR Physics + DSE]
- Why does signal equation include multiplicative T1 factor: (1 - e^(-TR/T1))??
  - Because that's the size fo the longitudinal component once subsequent pulses are applied! (What's left)
- Want to figure out derivation of double spin echo imaging equation

[T2 vs T2*] http://mriquestions.com/t2-vs-t2.html
- Spin echo: *S = G * PD * (1 - exp(-TR / T1)) * exp(-TE / T2)*
- T2* <= T2: Maybe can use as approximation for SE sequences?

[MRI BOOK (DEICHMANN)] + Look at page 587 for derivation
https://the-eye.eu/public/Books/Medical/texts/Handbook%20of%20MRI%20Pulse%20Sequences%20-%20M.%20Bernstein%2C%20et.%20al.%2C%20%28Elsevier%2C%202004%29%20WW.pdf
11.51: SPGR (In spoiling dont have angle == 90 degrees because want to cancel transverse components)

[MRI Book: IR equation derivations] http://www.cis.rit.edu/htbooks/mri/inside.htm

Also look at 4.11 in Green Bible

[FSPGR]
**Why is there an Inversion time (TI)??**
**NOTE** FSPGR (GE) == MPRAGE (Siemens)! (See Dave email 15.03.19)





# Log: 14.03.19
# Meeting
- Next steps: Use trained MPRAGE network to run inference on SABRE subjects: What parameter set minimises volume?
  - Assuming subjects have been acquired by protocol A at timepoint 1
  - Want to upgrade protocol: Acquire a single subject at same timepoint with new protocol
    - Want new protocol to be such that the difference in segmentations is as similar as possible
    - Can accomplish this by running inference on that subject, given the two protocol realisations: What value minimises volume?





# Log: 15.03.19 ##SABRE_EXPERIMENTS
- Running experiments under </home/pedro/SegmentationExperimentsNetworks/>
  - Physics_All_LRH: * Extended physics experiments with additional TI range 600 - 690 ms
                       * Due to the fact that SABRE data has TI = 650 ms
                       * If want to run inference want to have a network that has seen the TI
                       * Params: [STANDARD]
- Set up SABRE inference experiment
  - <data/SABRE/SABRE_res/TIs> TI slices
  - <data/SABRE/SABRE_res/Multiple_Realisations> Multiple copies of single subject for inference
  - <data/SABRE/SABRE_res/Inference> Inferred segmentations *TBC*

  - Started saving data to /data/ due to space constraints in SSD
  - Still need to set up randomly sampled SPGR
  - Still need to run Baseline on extended MPRAGE TI range

[2D TI slice notes]
From <sampler_grid_v2.py>: # When using 2D TI slice the network isn't able to find N equivalent image-like samples
                           # Since slice is homogeneous, have coordinates repeat N times to get around this:

                           if name != 'image':
                               coordinates[name] = np.tile(coordinates[name][0, :], (n_locations, 1))





# Log: Weekend
- Started running GIF segmentations for Extra acquisitions (TI = 600 - 690 ms)
- Stopped Physics_All_LRH





# Log: 18.03.19
Aims: - Monitor Baseline_All_Extra [Y]
      - Monitor GIF segmentations (Extra) [Y]
      - Run inference with many parameters on SABRE dataset _WIP_
      - Set up SPGR training <N>

[SABRE inference]
- Directory: <data/SABRE/SABRE_res/Inference>
- Use models saved in Physics_All_LRH for inference
**ERROR** Sampler tries to match spatial window sizes of images (same issue that was present with uniform sampler)
          When choosing a sampler you're also picking the grid sampler + aggregator for inference
          Make sure to redefine in application


**NOTE**
Need to WHITEN images prior to training/ as part of NiftyNet pre-processing
Should include brain stem as part of white matter (Otherwise hard for network to ascertain boundary + higher dice score) [BS == BrainStem] ##BS
- Include sentence in paper talking about how transfer to real domain requires extra care be taken in training
  - Whitening, noise, bias field
- Justification: Self-driving, prove that synthetic works in limited environment, prove that it generalises poorly, prove that style transfer is useful

[Physics_All_white]
**ERROR** LR was set extremely low due to copying of <ini> from [Physics_All_LRH] ~ 5e-5
            *Had reduced LR because was close to convergence*
          Rectified the issue: LR = 5e-4

[SPGR Params]
TR: 20 - 80 ms
TE: 4 - 10 ms
FA: 15 - 75 deg.

## __Don't forget to ask about OHE for mixed network__





# Log: 19.03.19
Aims: Generalising to real data

[SABRE Tests]
Running all inference tests under </data/SABRE/SABRE_res>

<Inference>             * Inference of whitening enabled network *POOR*
<Inference2>            * Inference of normalised SABRES (w/ + w/out whitening) *VERY POOR*
<Inference_white_bias>  * Inference of whitening + bias enabled network *AVERAGE on v2, POOR on v3*

Next steps: - Add noise to data (During simulation or augmentation?)
              - What level of SNR? *{15-20 Acceptable}, {20+ Good}*
            - Consider scaling bias augmentation further -> 1/2?

Add blurring (seg maths or scikit)
  Because of resamplingto 1mm when resolution could well have been worse beforehand
  *seg_maths -smo <std>* **SMOOTHER.sh**
Bias correct: 8 iterations should be enough
No good way to speed up training
Try turning off batch norm

[Noise implementation]
- Under <niftynet/engine/rand_noise.py>
  - Tried pyfftw, but no success: Multithreading poses issues
  - https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html#module-pyfftw.interfaces

> Currently running
Whitening + Bias + noise augmented network </home/pedro/Project/ImageRegression/3D_Noise_Test>
**NOTE** Had memory issues, reduced batch size to 2 and made sure to NOT save noise as separate array





# Log: 20.03.19 + 21.03.19
Aims: - Keep running network for Experiment 2
      - Try
      - **NOTE** Don't forget inference file: </home/.../ImageSegmentation/Scripts/inference_validation.py> ##INFERENCE ##VALIDATION

[Inference Tests]
Try for all subjects: Reduce range 600 - 1200 (13)

[Performance improvement notes]
sample_per_volume: 10 (Otherwise augment every iteration and throw away)
num_threads: 24 (Have 32 available)

Different relevant directories:
</home/pedro/Project/SegmentationNetworkExperiments> All networks related to Experiment 2
  <Physics_All> Initial test, LR = 2e-4                                                                            <WHITE><BIAS><NOISE><BLUR>
  <Physics_All_LRH> Same as previous, LR = 5e-4                                                                    <WHITE><BIAS><NOISE><BLUR>
  <Physics_All_white> Added image whitening (Otherwise scaling is an issue)                                        [WHITE]<BIAS><NOISE><BLUR>
  <Physics_All_white_Bias> Added bias field augmentation in addition to whitening (-0.5, 0.5)                      [WHITE][BIAS]<NOISE><BLUR>
  <Physics_All_white_Bias_noise_blur> Added noise augmentation (10, 18), used blurred images (seg_maths -smo 0.5), [WHITE][BIAS][NOISE][BLUR]
  increased bias (-1.0, 1.0)
  *mri_nu_correct.mni --i ${image} --o ${target_directory}/BC_${base} --n 8 --stop 0.0005*
  <Physics_All_white_LessBias_noise_blur> Reduced bias (-0.5, 0.5)                                                 [WHITE][BIAS][NOISE][BLUR]
  <Physics_All_white_LessBias_Lessnoise_blur> Reduced noise (8.0, 20.0)                                            [WHITE][BIAS][NOISE][BLUR]



Extra: * watch -n .1 nvidia-smi
       * htop




# Log: X
Aims: - SPGR simulations (Random Sampling + Pre-determined)
        - </data/Simulations/All_subjects_SPGR> Non
        - </data/Simulations/All_subjects_SPGR_RandomSampling> Random Sampling
      - Alter HRN-P accordingly (3 x 2 input params, therefore want at least 10 neurons)
        - Don't forget to consider OHE for mixed network, -> 15 neurons?

> Implemented mass_renamer_ini.sh
  Given a directory, will rename all files according to current directory name (where pattern == ini file)
  Tricks: * Can use <exec> after a command for nesting: [find ${directory} -type f -name '*.ini' -exec basename {} \;]
          * Filename (sans .ext) extraction: [${FILE%.*}] *Note that FILE in this case should be an assignment to actual filename*


# Meeting Notes ##SABRE_VALIDATION
- Experiment 2: Want to find combination of parameters such that volume difference is minimised
                - Can then argue that segmentation with the parameter pair is ideal for segmentation
                - Method: Pick 10 subjects, calculate 2D surfaces of parameter pairs for all (dice scores), find mean parameter pairs
                  - Extrapolate to other 12 subjects and verify their dice scores
      *Bridging done now* - Compare to GIF linear extrapolation: Find mean volumes between images (across N subjects), extrapolate linearly
                **FURTHER** Assume knowledge of one parameter (which we do): Find parameter for other such that volume is minimised
                              Find distance between "peaks" of this vs theoretical ideal

- Experiment 1: Keep architecture THE SAME: Don't want questions to be asked about choices
             2: Keep number of samples the same! (i.e.: Same MPRAGEs + SPGR)
             3: Argue that further work is needed to tweak hyperparams, architecture, etc.

Main bulk of work now: - Synthesise additional MPRAGE images to satisfy balanced requirement *_WIP_*
                       - Segment these using GIF *_WIP_*
                         - Don't forget about previous extras: </cluster/project0/possumBrainSims/SegTests/Data/Inf_Extra>
                  **NOTE** Don't forget that already have half the segmentations under </home/pedro/Project/GIF/AD_MPM/Segmentation_volumes/Inf_Labels>
                            (Excluding the aforementioned extras)
                         - Synthesise new parameter nifties for mixed network [Y]
                           - MPRAGE </home/pedro/Project/Simulations/Mixed_mprage_param_volume>
                           - SPGR </data/Simulations/Mixed_spgr_param_volume>
                       - Re-train MPRAGE + BASELINE <N>
                       - Re-train SPGR + BASELINE [Y]
                       - Train SPGR + MPRAGE + BASELINE <N>
                       - Inference MPRAGE should be good enough, can start running inference on all subjects and making plots <N>
                         - Need to set up directories accordingly (i.e.: Multiple Realisations)

[STANDARDISED PARAMETERS]
Iterations: 30000
Queue: 80
sample_per_volume: 16
LR = 5e-4
Loss: CrossEntropy
Regularisation: L2
Images per modality per subject: 121

[DGX-1]
TBD


# Log:
[Training + GIF]
- GIF segmentations: MPRAGE: 95%    </home/kklaser/Pedrinho/MPRAGE_121_extra_Inf> [KERSTIN_CLUSTER] _WIP_
                     SPGR: 85%      </cluster/project0/possumBrainSims/GIF/Inf> [CLUSTER] _WIP_

- Physics segmentations: [MPRAGE: 100%]  </home/pedro/Project/SegmentationNetworkExperiments/Physics_All_white_Bias_noise_blur> [LOCAL]
                         [SPGR: 100%]    </home/pedro/Project/SegmentationNetworkExperiments/RandomSampling_SPGR/models> [LOCAL]
                         [MIXED: 100%]     </home/pedro/Project/SegmentationNetworkExperiments/Physics_Mixed_121T> [LOCAL]

- Baseline segmentations: [MPRAGE: 100%]   </raid/pedro/Experiments/Baseline_MPRAGE_121T> [DGX1]
                          [SPGR: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/Baseline_SPGR_RandomSampling/models> [LOCAL]
                          [MIXED: 100%]  </home/pedro/Project/SegmentationNetworkExperiments/Baseline_Mixed_121T/models> [LOCAL]
                                         *Run locally due to significant amount of data involved and laborious external transfer process*

- Experiment 2 physics: [100% (Old)] </home/pedro/Project/SegmentationNetworkExperiments/Physics_All_white_LessBias_Lessnoise_blur> [LOCAL]
                        55% (New)    </home/pedro/Project/SegmentationNetworkExperiments/Physics_All_white_LessBias_Lessnoise_blur_121T> [LOCAL] _WIP_
                                     *Could have run on DGX1, but augmentation requirements suggest better suited for local training*


[Inferences]                                                 **[</data/Inferences_Final>]**
- Physics inferences: [MPRAGE: 100%] </home/pedro/Inference_Physics_MPRAGE_121T_outputs> [DGX1]
                                     </data/Inferences_Final/Physics_MPRAGE/Inf>         [LOCAL-FINAL]

                      [SPGR: 100%]   </data/SABRE/SABRE_res/Inferences_dir/Inference_RandomSampling> [LOCAL]
                                     </data/Inferences_Final/Physics_SPGR/Inf>                       [LOCAL-FINAL]
#                      MIXED: 0%

- Baseline inferences: [MPRAGE: 100%] </raid/pedro/Data/MPRAGE_121T_Inf>                   [DGX1]
                                      </data/Inferences_Final/Baseline_MPRAGE/Inf>         [LOCAL-FINAL]

                       [SPGR: 100%]   </data/SABRE/SABRE_res/Inferences_dir/Inference_Baseline_RandomSampling/Inf> [LOCAL]
                                      </data/Inferences_Final/Baseline_SPGR/Inf>                                   [LOCAL-FINAL]
#                       MIXED: 0%

- (Old) Inference: SABRE: [100%]  </home/pedro/Inference_SABRE_10_subjects_BC_outputs> [DGX1]


[Mixed network: Modifications]
- Physics parameters fed to network are in 3D [N x N x 10] array: [MPRAGE_OHE]
                                                                  [ SPGR_OHE ]
                                                                  [    TI    ]
                                                                  [    TR    ]
                                                                  [    TE    ]
                                                                  [    FA    ]
                                                                  [ exp(-TI) ]
                                                                  [ exp(-TR) ]
                                                                  [ exp(-TE) ]
                                                                  [  sin(FA) ]

- Had to remake all of the parameter arrays: [MPRAGE] - Same standard script with minor modifications
                                                      - <mprage_syn_all_subjects.py>

                                             [SPGR] - Had to create new script: Parameters sampled randomly therefore cannot reverse process as with MPRAGE
                                                    - Therefore create new script that reads exising param. arrays, extracts values and creates new array
                                                    - <spgr_param_extractor_and_param_slice_maker.py>

- Network related changes: Don't need to exponentiate on demand, therefore don't touch params further after reduce_max operations
                           Had to change grid_sampler to allow for 3D physics parameter passing at inference *GridSamplerPhysics*


**NOTE** ##BADSURFER
/usr/local/freesurfer/fsfast/bin/qsurfer: Changed name to badsurfer

[Dice scores]
https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
2* intersection / (sum of cardinalities)
- Found some IPMI code: Calculate dice losses individually for each tissue type, then average



Don't forget these things: - Linear correction on 10 subjects (GIF) [Experiment 2] *Not a priority*
                           - All things mixed experiments (26th)
                           - Actually check SPGR results (26th!) *Priority!*
                           - Set up training for SABRE 121T (i.e.: Blurred images) *5pm*

Plots: No standardisation vs linear standardisation vs Physics standardisation **Page 12 JOG**
Table: Coefficients of variation: https://en.wikipedia.org/wiki/Coefficient_of_variation <STD/MEAN> (Of the longitudinal images)


**ERROR** Different TDs used for MPRAGE simulations!!! -> Had used 0 originally, used 600 for (600 - 695, 705 - 1195 alternating)
Re-runnning simulations for 51 mis-simulated subjects: TI slices should be fine, however (1377 images!)

Affected networks: - MPRAGE Physics [FINISHED] (Inference ready to be queued on DGX1)
                   - MPRAGE_Baseline [*Running on DGX1 GPU 7*] (Inference awaiting models to be queued on DGX1)
                   - MIXED_Physics [*Running locally*]
                   - MIXED_Baseline [*Running locally*]
                   (Partially) SABRE inference (could always default back to originals)
Other: Need to redo GIF segmentations for these subjects *Have been submitted*

- Contour plot timing: ~20 mins per
- Blurs already made: Don't forget! **Low priority**

Overnight jobs: Mixed jobs + SABRE
                Also need to run inference on these jobs (-.-)
                  MPRAGE_Physics: Inference done
                  MPRAGE_Baseline: Inference done
                  Mixed_Physics: <NADA>
                  Mixed_Baseline: Being run atm

Main figures: - TBD


Monitor: Copying of LowTD to DGX1 _WIP_
         Making of blurred LowTD _WIP_
         Running of LowTD network **Almost done**
          Run inference immediately afterwards to look at percent variance

"Meta-analysis" paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5045042/pdf/nihms-808025.pdf
Results errors: /home/pedro/Project/ImageRegression/All_subjects_physics_inf/archive
                Related to sample_per_volume?? -> Running physics + baseline to investigate
                  Better but not the same, *Re-try with LowTD data*

Re-running LowTD networks with sample_per_volume = 1 (5k iters)
  Try running with 2D input slices inst

[Notes]
- Ultimate goal: Become invariant to site, sequence parameters, scanner

- Biomarker development: If can more consistently (PRECISION NOT accuracy) segment structures of interest, these become better predictors of disease

- POTENTIAL to exceed human performance: It is impossible for human to disentangle physics from phenotype: Humans only segment what they see according to contrast + intensities

- Ground Truth: Not important because PRECISION is what's aimed for, NOT ACCURACY, want to segment more consistently not find the ground truth

- Raters are flawed, can disagree, time consuming: This method promises to achieve greater consistency at lower cost



# Log: Post-MICCAI summary + organisation
To include: * Non-BN experiments (Currently running)
            * Organise all "Final" experiments
            * Organise Data: (Volume comparisons instead)
            * Missing GIF files: </home/pedro/Project/SPGR_GIF_Finale/Inf> *Can't login to cluster for some reason*
              * Run thi

[Scripts]
- New scripts: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/SABRE_inference_NoDice>: Physics
                                                                                               Percent volume differences between tissue types of v2 vs v3
                                                                                               Operating on: </data/Inferences_Final/SABRE/Inf>
               </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/SABRE_linear_standardisation>: GIF
                                                                                                     Percent volume differences between TT of v2 vs v3
                                                                                                     Operating on: </data/Inferences_Final/SABRE/Inf>

[SPGR redos] ##SPGR here
- Copied all segmentations into "OGs" folder: Can easily identify which files don't have a match
  - Copied all redos file names into .txt: SPGR_Missing_GIFs.txt <DamageControl>
  - Wrote small script that reads lines from .txt and copies files into directory <mver.sh>
  - Currently running GIF for these files under </cluster/project0/possumBrainSims/GIF/Inf/redos>

[SPGR_GIFs]
- "Finale": </home/pedro/Project/ImageSegmentation/GIF_SPGR/SPGR_GIF_Finale/Inf/inf_25>
  - All completed GIF segmentations of 121 SPGRs per subject
  -

[Slices]
- Running script to create MPRAGE training sets using slices instead of full 3D volumes
  - Using ALL slices overkill: sample randomly from each subject's simulations
  - Different folders for different number of slices: [30, 20, 10, 5]

[Images]
- Found under </home/pedro/MICCAI>
- WM + GM contour graphs, volume comparison plots, contrast comparison images





# Log: 11.04.19
[No Batch Normalisation experiments]
- Ran no BN experiment in </home/pedro/Project/SegmentationNetworkExperiments/Physics_MPRAGE_121T>
  - Running inference in </home/pedro/Project/SegmentationNetworkExperiments/Physics_MPRAGE_121T/Phys_Inference_noBN>

[Slice Experiments]
- Sampled all slices successfully: </data/MPRAGE_121_LowTD/SliceWorld/{}_slices>
- Label slices: </home/pedro/Project/GIF/AD_MPM/Segmentation_MPRAGE_121T/Train/Label_Slices>
- Param slices: </home/pedro/Project/Simulations/MPRAGE_Param_volume_Full3D/Train/TI_Slices>

> Experiments
- Running slice experiment (5 slices per sim) in </home/pedro/Project/SegmentationExperimentsNetworks/Physics_slice_test>
  *BS = 32, 1 epoch = 375*




# Meeting: To discuss
- BN tests: Good GM results, poor WM results
- Slice tests: Currently ongoing, better than base at GM, significantly worse at WM
- Changed HRN in preparation for baseline tests
- Consider shift to UNET?

# Meeting notes: to do
Training changes: - Change in GT: Choose middle range TI value instead of R1 segmentation [Y]
                  - Introduce augmentations: Bias field (minor, maybe ~0.2?) + Noise (Minor) <N>
                  - Work in 3D instead of 2D (64 cubed?) <N>
                  - Swap relu to lrelu (don't want zero at limits) <N>
[Others]
- Pick an image and pass multiple TIs to investigate if network using TIs or not
- Work on UNET implementation: Add parameters at end, turn off ALL BN

**[Moving forward]** ##NEWGIF
> GIF segmentations: </cluster/project0/possumBrainSims/NewSeason>
                     <MPRAGE_LowTD_0900>: Segmenting TI=900ms for training (Use instead of R1)
                      *Currently running*
                     <MPRAGE_LowTD_Inf>: Re-segmenting all inference volumes (LowTD, not sure about how compromised old data is)
                      *Currently running*

# Meeting: Additional things to try
UNet: Try larger patches (128 vs 64): Preferable to changing batch size
900 TI GT: Probably made a mistake in class argmax (Non-CSF labelled as CSF), just multiply by smoothed TIV
Loss: Try Dice + X-Entropy: Dice more aggressive -> Faster convergence *"DicePlusXEnt" in NiftyNet*
  Largest connected component -> Dilation -> Fill -> Erosion
  [seg_maths <INPUT> -bin -lconcomp -dil 1 -fill -ero 1 <OUTPUT>]

Argmax issues: Should only argmax in regions where sum of tissues is at least 0.5 -> This is why CSF excess problems arise

Regularisation notes: http://www.chioka.in/wp-content/uploads/2013/12/L1-vs-L2-properties-regularization.png
UNet requirement: Subtract by 32 divisible by 8

# 18 - 26: Easter break





# Log: 29.04.19
Aims: Check jobs that were left running last time
      Keep implementing changes discussed at previous meeting

[Issues from last time]
- No compatibility for int64 (?) in seg_maths, require conversion

[Things to remind myself of]
- New ground truth segmentations (Based on 900ms): Completed? Already using?
  - **Already completed**: </home/pedro/Project/GIF/AD_MPM/0900_Argmax/Masked_argmax>
  - Largest connected component -> Dilation -> Fill -> Erosion (Get rid of unconnected voxels, dilate then erode to promote smoothness)
    - Currently being used in HRN + UNet experiments

[Experiments to check]
- UNet 3D MPRAGE: On DGX1, transferred files locally to </home/pedro/Project/SegmentationNetworkExperiments/UNet_Baseline_MPRAGE_3D> *60k*
  - Ran inference on DGX1 under same folder + <Inference>: Segmentations seem good for low TIs, awful for higher TIs
  - Re-running w/ addition of noise (8.0, 20.0) + bias (0.2)
- HRN-Physics 3D MPRAGE: </home/pedro/Project/SegmentationNetworkExperiments/Physics_MPRAGE_3D> *30k*
  - Couldn't run inference for some reason, repeating experiment w/ noise (12.0, 20.0) + bias (0.2)
  - Had issue relating to incompatible types: Expecting float but getting double
    *0-th value returned by pyfunc_0 is double, but expects float*
    [FIXED] Relates to noise augmentation: Noise was being added as a double, altering the image type to double
            Easy fix: Convert to float32 using ...astype(np.float32)

[Amigo meeting]
- Plan: Train networks in 3D
        UNet implementation
        Uncertainty analysis
- Other: Help Artiza to develop his MR simulator code
         Update research log





# Log: 30.04.19
Aims:

[Less data approach]
Run UNet with half data: Same range, sparser </raid/pedro/Labels/0900_Labels/500s>
                         Same density, less range </raid/pedro/Labels/0900_Labels/HalfRange>

[Experiments currently running]
Baseline UNet 3D <
Baseline UNet 3D + HalfRange <Running in tmux pane 1>
  - Seems to be performing relatively well *Clear decrease in grey matter segmentation with increasing TI* <Inference running in tmux pane 0>
  - Extend to Physics to observe performance? -> **NOTE** Stop training in tmux 1, queue physics training to gain some preliminary analysis
Physics UNet 3D
Physics HRN 3D

[Inference]
3D UNet performs extremely poorly, even with augmentations: Best performance for ~ 700ms
  Little change when changing inference parameters *General rule: Divisible by 8 & 128 -> 40*





# Log: 01.05.19
Aims: Check on progress of various networks
      Relay introductory work in Jorge meeting

[Experiments currently running]





# Jorge meeting
Points made: - Baseline UNet performed poorly in regime of large TI range (600 - 1200ms)
               - Better performance achieved when limiting TI range (600 - 1000ms)

Feedback: - Network could have issues relating to non-normalised intensities
            - [Whitening]
          - Vald nature of vanilla 3D UNet means that output is cropped
            - ['Same' padding where border == loss border to maximise region of computed loss]

Border: Stipulated receptive field border at inference time
Loss border: Defines "crop" border when calculating loss, i.e. regions of image to ignore (e.g.: Padding that contains no information)





# Log: 03.05.19
[Notes on matrix changes with convs/ pooling]
Convolutions: (W − F + 2P)/ S + 1
Pooling: (W - F) / S + 1
<W>: Input volume size
<F>: Kernel size
<P>: Padding
<S>: Stride

> Valid padding: Output = CEIL((Input - Filter * Dilation)/ Stride)
> Same padding: Output = CEIL(Input / Stride)


Tracking UNet concats:
Same (132 base):

Valid (128 base):

[Experiment Notes]
> Inferences
OLD (Bad implementation) of UNet 3D + Physics returns full sized segmentations at inference time for some reason
  - **NOTE** Add to volume padding size at inference time! (If inference border == x, pad by x/2)

[Experiments in progress log]
- New UNet implementation with SAME padding, whitening   __Baseline__*i.e.: 128 -> 128*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_3D_Experiments_HalfData>
  - Running in <PANE-0>

- Old UNet implementation, VALID, whitening              __Physics__ *128 -> 40*
  - </raid/pedro/Experiments/UNet_MPRAGE_Physics_3D_Experiments_FullRange_white>
  - Running in <PANE-3>

- Inference of Old UNet, VALID                           __Physics__ *128 -> 40*
  - - </raid/pedro/Experiments/UNet_MPRAGE_3D_Experiments_Physics/Inference>
  - Running in <PANE-4>

**NOTE** Need to run FULL data experiment with baseline!


## OTHER
Blogs to read:
BN: https://towardsdatascience.com/intuit-and-implement-batch-normalization-c05480333c5b
Network checklist: https://karpathy.github.io/2019/04/25/recipe/
Optimisation algorithms: https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms


Gish Gallop
BS Asymmetry





# Log: 07.05.19
Aims: - Run inference on finished experiments [Inference]
      - Investigate differences between 'same' vs 'valid' training
        - Investigate differences between physics vs baseline [Analysis]
          - Potentially re-use volume comparison script used for MICCAI work

[Inference]
- Old UNet, VALID, whitening, Half (Misnamed!!)          __Physics__ *128 -> 40*
  - </raid/pedro/Experiments/UNet_MPRAGE_Physics_3D_Experiments_FullRange_white>
  - Running in <PANE-1>

- New UNet, VALID, whitening, half                       __Baseline__ *128 -> 128*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_3D_Experiments_HalfData>
  - Running in <PANE-0>

**NOTE** Need to find wherabouts of GIF segmentation
  - Had foresight to do this: </cluster/project0/possumBrainSims/NewSeason/MPRAGE_LowTD_Inf> *Cluster*
  - Copying GIF segmentations to </home/pedro/Project/GIF/AD_MPM/Segmentation_volumes/GIF_Inf_121> _WIP_
    - **Missing** a single segmentation: Subject 6, TI = 0.69s: Re-running on cluster
    - **Don't** forget to argmax to create standard 3D segmentation

[Analysis]
- Script: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/MPRAGE_inference_validation_paper.py>
  - Plots volumes across TI range centered on zero

Overall: GIF best in WM + GM
         Physics very best in CSF
         GIF particularly poor in TIV

[New Experiments]
- Full data range + normalisation, baseline + physics
  - Running in <PANE-0> *Baseline*
  - Running in <PANE-2> *Physics*





# Log: 08.05.19
Aims: Check experiments in progress
      Implement meeting feedback

# Jorge meeting
GIF vs Base vs Physics: Promising, sanity check that Phys perf. > Base perf.
                        Seems odd that linear relationship is observed in Phys: Network should be able to account for it
                          - Freeze UNet excluding physics branch: Can tell that segmentation performs well, just need to ameliorate physics component
                          - Basic network changes: RMSProp: Erratic nature of gradients (due to BS of 1) means that adam is not as useful
                                                            Reduce patch size to allow for larger batches (See above) *Use loss border parameter for semi-valid training approach*
                          - Architecture changes: Concatenate + add another 1D convolution
                                                    By concatenate mean add it further along in the network as well
                                                  Consider adding physics branch elsewhere in network

[Augmentations]
**ERROR** Was adding bias & noise to <NORMALISATION> instead of <AUGMENTATION> layers
  Could have an effect on how things are ordered: Currently have ambiguous ordering in that whitening + noise + bias are bundled together

[Gradient freezing]
Line in physics_segmentation_application:
                                  grads = [(grad, var) for grad, var in grads if 'FC' in var.name]
- With this syntax, only 'FC' variables will be updated

[Experiments]
- New UNet, VALID, whitening, Full                       __Baseline__ *132 -> 44*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_3D_Experiments_FullRange_white>
  - Running in <PANE-0>

- New UNet, VALID, whitening, Full                       __Physics__ *128 -> 128*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_Physics_3D_Experiments_FullData>
  - Running in <PANE-2>

- New UNet, SAME, whitening, Full, BS 3                  __Physics__ *96 -> 96 + loss border 22*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_Physics_3D_Experiments_FullData_BSLarge>
  - Running in <PANE-1>


[Next time] - Frozen gradients test
            - Increased complexity test of FC layers
            - RMSProp?





# Log: 09.05.19
Aims: - Check on three running experiments
        - Run inference if necessary
      - Pycharm deployment feature (Dan + Tom)
      - Registration for MIDL

[Experiments + Inference]
- "Full range" increased BS experiment is actually half range!
- "Full range" standard BS experiment is half range!
- "Full range" Baseline experiment is **CORRECT**
- Need to run equivalent for Baseline to compare





# Log: 10.05.19
Aims: - Check on experiments
        - Evaluate inference
      - MIDL registration
      - Frozen gradients test

[Bash scripts]
New addition: **inf_organiser.sh**
                - Creates subdirectory structure of inference folders
                - Moves inference images into corresponding folders

[Experiments]
Running Baseline + Physics for Full Data, batch size of 3 <PANE-2> + <PANE-1>
Running Baseline for Half Data, batch size of 1 for comparative purposes <PANE-0>

- New UNet, semi-VALID, whitening, Full, BS 3            __Baseline__ *96 -> 96 + loss border 22*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_3D_Experiments_FullData_BSLarge>
  - Running in <PANE-2>

- New UNet, semi-VALID, whitening, Full, BS 3            __Physics__ *96 -> 96 + loss border 22*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_Physics_3D_Experiments_FullData_BSLarge>
  - Running in <PANE-1>

- New UNet, semi-VALID, whitening, Half, BS 3            __Physics__ *96 -> 96 + loss border 22*
  - </raid/pedro/Experiments/New_UNet_MPRAGE_3D_Experiments_HalfData_BSLarge>
  - Running in <PANE-0>





# Log: 13.05.19
Aims: - Check on experiments
      - MIDL registration
      - Frozen gradients test

[Experiments]
All trained well, though noticeable overfitting exhibitted in both Basline + Physics for full data
  However, validation does not seem compromised, so can just use -1 for inference

[Inference]
Running inference on ALL experiments listed above, in same panes

[Inference Plot conclusions]
Half data: - No significant quality improvement over base
Full data: - Significant quality improvement over base
           - Parabola-like behaviour: Suggests physics parameters truly affecting segmentation quality
           - Major improvements in <WM> + <CSF>, slightly worse performance than base in <GM>
             - Suggests that network going for "overall" segmentation improvement rather than treating classes individually

Branch-off work: - Take converged network and train with all but physics gradients frozen
                 - Run inference for same point in both Baseline + Physics as sanity check
                 - Run experiments with increased network complexity -> Increase ability of network to utilise PP as it sees fit

[Other]
Successfully modified <inference_constructor.sh> s.t. it can use script variables in vim edit calls (See ##vim --quotes)
Successful ...





# Log: 14.05.19
Aims: - Harmonisation literature review
      - Experiments monitoring
      - Find SABRE inference experiments (VIP!)

[Harmonisation literature]
A Lifelong Learning Approach to Brain MR Segmentation across Scanners and Protocols: https://arxiv.org/pdf/1805.10170.pdf
Summary: - Want to harmonise segmentation across protocols/ scanners
         - Can leverage domain specific BN parameters evolved from mono-batch domains
           - i.e.: D1 -> bn1, D2 -> bn2, D3 -> bn3 etc.
         - Want to generalise to new domain with few training examples: Evaluate on all available BN params and see which one performs best
           - Train network initialised on best BN parameters (e.g.: bn2)
           - This was shown to perform almost as well as networks trained

[SABRE]
Reminder: 22 subjects
          Paired MPRAGE scans: Differing protocols

Experiments: Trained on (blurred) synthetic </home/pedro/Project/SegmentationNetworkExperiments/Physics_All_white_LessBias_Lessnoise_blur_121T>
              Ran inference on 10 subjects </raid/pedro/Experiments/Inference_SABRE_10_subjects_BC>

Moving forward: Run inference using most recent BS3 Physics models found in: </home/pedro/Project/SegmentationNetworkExperiments/New_UNet_MPRAGE_Physics_3D_Experiments_FullData_BSLarge>
  - Result: Not particularly good, major gaps in segmentations
      Blur + Noise probably necessary

[Frozen gradients]
- Resumed from 32519 (Full gradients)-> Evaluating at 45227 (Physics gradients only)

[Other]
- Consider doing histograms: Investigate information content across TI ranges, subjects, modalities





# Log: 15.05.19
Aims: - Investigate Frozen Gradients inference + compare against default
      - BN paper: To try to implement or not?

[Lifelong approach implementation details]
- Ensure each batch contains only samples from a single modality
- Start of training: Initialise all batch parameters randomly
                     Pass in batch of modality k -> batch parameters k update -> freeze batch parameters k
                     Pass in batch of modality k + 1 -> batch parameters k + 1 update -> freeze batch parameters k + 1
- Could have array where each row corresponds to batch parameters of specific modality
  - Train modalities cyclically: A -> B -> C -> A -> B -> C -> A ... etc.


# Jorge meeting: Notes
- Ender paper: * Limited use
               * Only show that method works over proven very bad method (i.e.: Inference on unseen modality is predictably awful)
               * No validation on common sense method of just training on new modality w/ augmentations: No reason this wouldn't work well
               * Can only account for linear differences between modalities (bec. BN), no non-linearity
               *Could consider small test whereby network is trained to certain point then everything but batch norm is frozen when training on new modality COMPARED to just vanilla network*

Future work: * Look into probabilistic UNet paper: https://arxiv.org/pdf/1806.05034.pdf  _WIP_
              **NOTES** Seems complicated, dedicate time tomorrow to go through more thoroughly
                        In meantime invesitgate Monte Carlo dropout (As simple as setting keep_prob to non-one value at Train + Inf.)
             * Can look into implementing part of Ender based on physics parameters  <NotWorth>
             * Augmentation: Now that network overfits (hence has enough capacity) can consider rotation augmentations  [Y]
               *Probably will be beneficial when running inference on SABRE dataset*
             * Architecture work: Best if can load model of semantic part of network (i.e.: Main body of UNet)
                                  **Talk to Tom about feasibility of implementation**

[New Experiments]
Physics Blur with <rotation> (-> SABRE)
Physics non-Blur with <dropout> (-> Uncertainty)
**NOTE** Monitor "Complex Architecture" experiment: Not all classes being represented in output for some reason
         Problem seems to persist even when "downgrading" to simpler (proven) architecture *Currently verifying this*





# Log: 16.05.19
Aims: - TBD

[Custom architecture problem]
- Want to load pre-trained network weights into multiple custom architectures
- Could have custom application where output of one network feeds into second ('physics') network
- <HOWEVER> for partial loading of model this approach will not work: Not easy to do w/out keras

[Error note] ##ERROR
IF <Assign requires shapes of both tensors to match. lhs shape= [192,5] rhs shape= [2,5]>: Missing physics_spatial_window_size





# Log:17.05.19
Aims: - Physics Monte Carlo checks + plots
      - Baseline Monte Carlo set in motion
      - Complex architectures experiment check

[Experiments]
- Checking on Monte Carlo inferences (11)
- Running Baseline Monte Carlo inference (21)





# Log: 20.05.19
Aims: - Check inferences: Baseline dropout + Physics SABRE inference
      - Investigate Complex Arch

[Multi-architecture experiments]
- Created modified unet, unet_physics_same_flexible
  - By passing a flag, "conc_flag", physics parameters get concatenated at different stages in the network
    -1, -2, -3: Downsampling branch concatentation (Numbers correspond to level)
             0: Vanilla/ Baseline
    +1, +2, +3: Upsampling branch concatenation (Numbers correspond to levels)

> Experimental setup brainstorm
- Want to train a separate network for each concatenation possibility
- Therefore need to pass a different conc_flag value while maintaining all other network parameters

- Need for loop that: 1. Creates experiment subdirectory
                      2. Call setup.sh of template .ini and .sh
                        2b. Ensure that <concFlag> is changed according to for loop variable
                      3. Runs experiment *careful with GPU allocations! {0, 1, 5, 6, 7}*
             **NOTE** No need to run experiments for <concFlag> == 1 since this is the "default" concatenation

> Experiments
- Started running: </raid/pedro/Experiments/Multi_UNet_MPRAGE_Experiments/Conc_flag_-1_experiment>
                   </raid/pedro/Experiments/Multi_UNet_MPRAGE_Experiments/Conc_flag_2_experiment>
                   </raid/pedro/Experiments/Multi_UNet_MPRAGE_Experiments/Conc_flag_-2_experiment>

[Ongoing issues]  ##NOISEBREAK
Noise augmentation: SNR "rule of thumb" where 20+ is generally considered to be good not holding
                    When <whitening> on even 30 leads to inability to train





# Log: 21.05.19
Aims: - Dropout investigation **Meeting with Zach at 3 about this**
      - Diversified network investigation

[Dropout methods]
- Standard method: Activation dropout: Kernel is applied, THEN dropout applied on activations
- Kernel dropout method: Weights in kernel dropped out, THEN convolution happens
  - Implementation: https://github.com/apratimbhattacharyya18/seg_pred/blob/master/weight_dropout.py
  - See also: https://openreview.net/forum?id=rkgK3oC5Fm *Bayesian prediction of future street scenes using synthetic likelihoods*

[Dropout details]
NiftyNet: Keep prob not exposed in most networks (AFAIK)
          Manually exposed according to implementation above

> Where to dropout
Bayesian SegNet: https://arxiv.org/pdf/1511.02680.pdf ##DropoutSeg <>
  - Immediately following pooling, immediately before upsampling (6 instances in UNet3D therefore)





# Log: 22.05.19
Aims: - Check on multi-architecture experiments
      - Monitor dropout experiments
      - Begin implementation of alternative (kernel-based) dropout
      - Test noise augmentation (Set noise to "zero" as first approach)

[Multi-architecture]
**ERROR** Was using <FROZEN> physics segmentation application instead of standard one
            Therefore most network gradients are NOT UPDATED
            Have rectified the issue and set all experiments running (including conc_3 == standard concatenation protocol)

[Skull stripping + mask]
- Employed to circumvent issues associated with fat hyperintensities
> HD-BET: https://arxiv.org/abs/1901.11341
HD-BET: <GPU> hd-bet -i <INPUT> -o <OUTPUT>

[To-do list]
- Debug noise augmentation <N>
- Implement kernel dropout [Y]
- Use skull stripped data in new network
  - Follow up with SABRE inference





# Log: 23.05.19
Aims: - Kernel dropout implementation (Read up on log likelihood + log. reg again!)
https://en.wikipedia.org/wiki/Likelihood_function
https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf
      - Check on skull stripped data generation
      - Monitor modified UNets performances

[Notes]
Likelihood: Expression of likelihood of function parameters given observation of variables
Log-likelihood: Speed, numerical stability, simplicity
Individual labels ~ Bernoulli distribution (Special case of Binomial where n=1)

>Maths
Sigmoid(log(x/(1-x))) == x
FULL kernel shape == (BS, N, N, inpFeats, outChns)

[Kernel dropout]
- Implementing it in <NiftyNet-5/../niftynet/layer/convolution.py> *Kernel + bias initialised here*
  - Follow mask approach: 1. Generate mask of same shape as kernel
                          2. Set mask elements to zero according to keep_prob





# Log: 28.05.19
Aims: - Kernel dropout experiment check
      - Skull stripped data experiments: Begin these
      - Skull stripping: Complete on inference data as well + SABRE data
      - Monitor modified UNets
      - Investigate/ Reminder: * Standard Dropout experiment </raid/pedro/Experiments/New_UNet_MPRAGE_Physics_3D_Experiments_FullData_BSLarge_Dropout>
                               * Kernel Dropout: Did implementation work?
      - Probabilistic UNet: More in depth reading

[Skull-stripping]
- Train images completed
- Running on inference images </raid/pedro/Data/SS_MPRAGE_LowTD_121_Inf>

[Probabilistic UNet] https://arxiv.org/pdf/1806.05034.pdf
- Reasoning: - Instead of having point sample for imaging physics want to sample from latent space in such a way that uncertainty is captured
              - E.g.: For extreme TI values want network to learn distribution over parameter values

[Ongoing To-do list]
- Debug noise augmentation                          _WIP_

- Standard dropout tests: * Physics Train            [Y]  </raid/.../New_UNet...Dropout>
                            * MC dropout at TT      _WIP_
                          * Baseline Train           [Y]  </raid/.../New_UNet_MPRAGE_3D_Experiments_FullData_BSLarge_Dropout/Inference>
                            * MC dropout at TT      _WIP_

- Implement kernel dropout                           [Y]
  - Train network with Kernel Dropout                [Y]
    - MC dropout at TT                               [Y]  </raid/.../Monte_Carlo_Inferences_KD>

- Finalise skull-stripping: * Training data          [Y]
  *All folders have*        * Inference data         [Y]
  *a SS_ prefix*            * SABRE inference data   [Y]
                            * Blurred Training data  [Y]

- Use skull stripped data in new network             [Y]  </raid/.../SS_New_UNet_MPRAGE_Physics_3D_Experiments_FullData_BSLarge>
  - Run inference on this network                    [Y]

- Use blurred skull stripped data in new network    _WIP_
  - Follow up with SABRE inference                   <N>

- Probabilistic UNet understanding + implementation _WIP_

- Modified UNet evaluations                          <N>

[Bin additions]
> dgxboard.sh <LocalPort> <DGXPort>
- Creates ssh tunnel to allow for tensorboard to be called on logs on the DGX1
  - Opens tensorboard in the browser

Older, previously undocumented files
> skull_strip_dir.sh <ImageDir> <OutputDir> <GPUID> *Only on DGX1*
- Submits skull-stripping job on all nifties in specified directory by calling HD-BET https://arxiv.org/abs/1901.11341

[Noise augmentation: Debugging]
- Set noise to "zero", see what happens
- Reverse normalisation and augmentation pre-processing (In application file, process is sequential)





# Log: 29.05.19
Aims: - Probabilistic UNet in depth reading
      - Monitor KD inference (Kernel Dropout)
      - Start SD inference

[Probabilistic UNet: continued]
- Implementation: https://github.com/SimonKohl/probabilistic_unet/blob/master/model/probabilistic_unet.py
- Questions: * What is the latent prior space? What does it mean to sample from it?
               1. During training it is pulled "closer" to posterior distribution *Pull each other closer together*
               2. During inference, each input produces a different latent space
                 2a. Can then sample from latent space to get samples that get concatenated to late UNet activations
             * How does the posterior network work (Know that it takes GT + raw input and maps to {mu_post, sig_post})?
               * How is it trained?
               *Posterior network is an encoder, taking seg + inp to create a downsampled latent representation of these*





# Log: 30.05.19
Aims: - P-UNet discussion *See Jorge if possible*
      - KD inference plots

[KD inference plots]
**NOTE** Running out of SSD space, <20 Gb remaining *Fixed for now, ~60 Gb free*
- Plots in progress

[Papers/ other]
> Multi-sample dropout: https://arxiv.org/pdf/1905.09788.pdf, https://twitter.com/karpathy/status/1133931772855537664
- Idea is to sample M times at dropout == M-plication of minibatch size w/out additional computation cost
  - Sample dropout masks following conv + non-lin. -> Duplicate weights -> Average -> Loss
  - Operations only duplicated after dropout, therefore just sampling masks rather than conducting full forward pass

> Other useful resources: http://www.arxiv-sanity.com/
                          https://www.youtube.com/user/keeroyz/videos
[Arxiv Sanity] Collection of newest papers in field, ease of perusing through submissions + picking out relevant entries

> P vs NP: https://en.wikipedia.org/wiki/P_versus_NP_problem
           https://arxiv.org/pdf/1905.11623v1.pdf
The general class of questions for which some algorithm can provide an answer in polynomial time is called "class P" or just "P". For some questions, there is no known way to find an answer quickly, but if one is provided with information showing what the answer is, it is possible to verify the answer quickly. The class of questions for which an answer can be verified in polynomial time is called NP, which stands for "nondeterministic polynomial time".
> Subset sum problem (NP-hard): https://en.wikipedia.org/wiki/Subset_sum_problem
> Turing machines: https://arxiv.org/pdf/1904.09828v2.pdf MTG





# Log: 31.05.19
Aims: - Noise augmentation debugging
      - Blurred training monitoring

[Noise debugging]
- Try swapping normalisation & augmentation (Augmentation -> normalisation)  ##SWAPS
  - Reasoning: Noise addition FOLLOWING normalisation means that values are not constrained

[SS Blurred training]
- Had early issues with Nans in summary histogram: Realised had mixed up noise/ bias parameters *Bias had been set to high integer values*
  - Training with swapped normalisation & augmentation *Network seems to train just fine for now*





# Log: 03.06.19
Aims: TBD

[Experimental tracking frameworks]
- Wandb: https://www.wandb.com
- omniboard: https://github.com/vivekratnavel/omniboard





# HACKATHON
Made an absolute meal out of tensorflow installation, gist of it is:
- Conda installation messes with location of python + python libraries
- Could not run python/ tensorflow from anything other than conda envs
- Need TF 1.10!!! pip3 install tensorflow-gpu==1.10
- Python 3.6
- numpy 1.16.1




# MICCAI reviews:  ##REVIEW

- Major weaknesses identified: * GMM (GIF)for ground truth not seen as suitable, not similar enough to "true" human-like segmentations *Medium*
                               * Lacking in evaluation metrics: Why just CoV? Dice, Volume difference, MSD, HD <Hard>
                               * Lacking in references <Hard>
                               * Lacking in simulation method description, only described as "simple and robust" [Easy]
                               * Lacking description of SABRE dataset  [Easy]
                               * Don't justify simulation parameters  *Medium*
                               * Figure description lacking  [Easy]
                               * Lacking limitation discussion  *Medium*

- Strengths: * Novelty of physics method
             * Synthetic data relevance
             * Clinically motivated
             * Robust method

Gaussian mixture modelling based on MPMs: Find average literature values for tissues, pick some sigma, create maps based on these
  - Find files: </home/pedro/Project/Simulations/AD_dataset/MPMs/PMs>
  - Create three gaussians: One for each tissue type
    - Set mean + standard deviation to literature value of these, already have a few papers to go by:
      https://cds.ismrm.org/ismrm-2001/PDF5/1391.pdf (3.0T, all tissues [STD], comparison against others <NoSTD>)
      http://www.mri-q.com/uploads/3/4/5/7/34572113/field_dependence_relaxation.pdf (Many T not including 3.0T, mention some papers that do [STD])
        https://onlinelibrary.wiley.com/doi/epdf/10.1002/%28SICI%291522-2586%28199904%299%3A4%3C531%3A%3AAID-JMRI4%3E3.0.CO%3B2-L (3.0T, no CSF [STD])
        https://onlinelibrary.wiley.com/doi/epdf/10.1002/1522-2594%28200101%2945%3A1%3C71%3A%3AAID-MRM1011%3E3.0.CO%3B2-2 (3.0T, no CSF [STD])
        https://link.springer.com/content/pdf/10.1007%2Fs10334-008-0104-8.pdf (Many T1)
    - Assign probabilities on pixel level for each tissue
    - Consider skull stripping (Non-zero T1 values outside skull might interfere)
Potential values: WM: [838 +/- 78] [847 +/- 43] 3.0T
                  GM: [1283 +/- 161] [1763 +/-60] 3.0T
                  CSF:                           [4472 +/- 85] -> 4.0T

> Theory
T1 dependence on Magnetic Field Strength: http://mriquestions.com/bo-effect-on-t1--t2.html
T1 relaxation: http://mriquestions.com/what-is-t1.html

Argue for a physics gold standard + precision! Drive home this point
Ground truths do not exist in image segmentation. The closest gold standard that can be attained relies on contrast which is subject to protocol and scanner variabilities.





# Log: 13.06.19
Aims: Physics Gold Standard (PGS) work

[Physics Gold standard]
Gaussian parameters: Had to fudge the numbers somewhat: * cGM: Fairly close
                                                        * CSF: Exact (Large STD)
                                                        * WM: Significant deviation: 840 -> 1040

Eroding 600ms masks (from [</home/pedro/Project/Simulations/AD_dataset/MPMs/PMs/Masks_600>]):
> for f in SS_MPRAGE_sub_*; do seg_maths $f -ero 1 ../Ero_Masks/Ero_$f; done

Erosion segmentations creation process (from [</home/pedro/Project/Simulations/AD_dataset/MPMs/PGS>]):
> for ID in {0..26}; do fslmaths Argmax_BP2_PM_MPM_sub_${ID}.nii.gz -mul ../PMs/Ero_Masks/Ero_SS_sub_${ID}_mask.nii.gz SS_PGS/Ero_Argmax_PM_sub_${ID}.nii.gz; done


[T2* considerations]
Harder to find literature values on this topic, but advantage is that there is no MFS dependence
Examples: http://www.ajnr.org/content/29/5/950/tab-figures-data
Potential values: WM: [67.6 +/- 11]
                  GM: [48.5 +/- 12.1]
                  CSF:               [500 +/- 200]

[PD considerations]
Proton densities are relative





# Log: 14.06.19
Aims: Literature aggregation for MPM creation

[MPM creation: Literature]
- Should do semi in depth study of ALL MPM (specifically T1) papers
- Variability in claimed T1 values varies **SIGNIFICANTLY** (1200 - 1800 for GM, 790 - 970 for WM) from paper to paper
  - Try to find paper whose MPM protocol most closely follows YOAD (i.e. My Young AD dataset)
    *See YOAD method description here: </home/pedro/Project/Simulations/AD_dataset/YOAD_MPM_protocols.pdf>*

[YOAD protocol]
> "Images for the YOAD study were acquired on a 3T TIM Trio whole-body MRI scanner (Siemens Healthcare, Erlangen, Germany) at the National Hospital for Neurology and Neurosurgery, Queen Square (UCLH NHS Foundation Trust) using a 32-channel receiver array head coil.""

<T1W> * TR = 24.50ms
      * TE = 2.2 -> 14.3 ms (MT + R2*), TE = 2.2 -> 19.7 (R1)

>> CONTINUE HERE ##
Conversion to probabilities: 1. Merge all tissue "fake-probs" -> 4D <MergedOriginal>
                             2. Sum them all together -> 3D <MergedSum>
                             3. Merge sum multiple times to be same size as <MergedOriginal>
                             4. Divide <MergedOriginal> by <MergedSum> to get probabilities

> Combined command: *Just for subject 13, will generalise after properly tweaked tissue dists.*
seg_maths cGM_PM_MPM_sub_13.nii.gz -merge 2 4 WM_PM_MPM_sub_13.nii.gz CSF_PM_MPM_sub_13.nii.gz merged_13.nii.gz; seg_maths merged_13.nii.gz -tmean -mul 3 sum.nii.gz; seg_maths sum.nii.gz -merge 2 4 sum.nii.gz sum.nii.gz -recip -mul merged_13.nii.gz fin_merged_13.nii.gz


# Log: ...  #VIP
Things to address: - Probabilistic map creation
                     - Skull-stripped version: NOT HD-BET
                     - Seg_stat issues
                     - Reference papers
                   - CMIC Summer school todos
                   - Argmaxing: Mention re-arranging of parametric maps, splitting of existing 4D prob maps
                   - Experiments: BS1, BS3, SS_BS3 *Physics + Baseline*
                     - Initial issues: What loss to use: Used RMSE initially but transitioned to dense dice
                     - Implementation: Had to iterate for RMSE, but dense dice support exists out the box
                     - Is SS actually the best approach??
                   - Bias field augmentation: Short explanation/ recap
                   - Blurring: Keep checking process


> Probabilistic map creation
[06.11.19] Creation script: </home/pedro/Project/ImageSegmentation/Scripts/physics_gold_standard.py>
[06.11.19] MPRAGE ProbMap Labels: </home/pedro/Project/Simulations/AD_dataset/MPMs/PGS_T1s/SS_prob_maps>

dmesg -T| grep -E -i -B100 'killed process'
find -type f -exec mv -v {} . \;
for f in Label_*; do seg_stats $f -r; done
call uthr with 1.1

Legendre polys: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=845174
We propose to estimate the bias field by a parametric
model. Because the bias field is usually very smooth across
the whole data set, we assume that belongs to a family of
smooth functions. We have chosen to be a linear combination of smooth basis functions

for f in Back_Dil_fin_merged_sub_*; do fslsplit $f ./Argmax/split_${f} -t; done

for num in $(seq 0 26); do fslmerge -t PropMerge_${num}.nii.gz split_Back_Dil_fin_merged_sub_${num}0003.nii.gz split_Back_Dil_fin_merged_sub_${num}0002.nii.gz split_Back_Dil_fin_merged_sub_${num}0000.nii.gz split_Back_Dil_fin_merged_sub_${num}0001.nii.gz; done

for f in PropMerge_*; do fslmaths $f -Tmaxn Argmax_${f}; done

Don't forget with "New" Skull Stripped data: Using very eroded GIF TIV masks NOT HD-BET





# Log: 26.06.19
Aims: - Monitor all experiments
      - Start on/ finish CMIC summer school poster
      - Plot plot comparison plots by end of day (hopefully)

> Experiments
[Training]

- Physics segmentations: [BS1: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Physics_MPRAGE_3D> [LOCAL]
                         [BS3: 100%]    </raid/pedro/Experiments/BS3_ProbMaps_Physics_MPRAGE_3D> [DGX1]
                         [SS_BS3: 100%]   </raid/pedro/Experiments/SS_ProbMaps_Physics_MPRAGE_3D> [DGX1]

- Baseline segmentations: [BS1: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Baseline_MPRAGE_3D> [LOCAL]
                          [BS3: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Baseline_MPRAGE_3D> [LOCAL]
                                        *Same folder as BS1: Seemed unnecessary to make a new folder for such a small change*
                          [SS_BS3: 100%]   </raid/pedro/Experiments/SS_ProbMaps_Baseline_MPRAGE_3D> [DGX1]

**NOTE**
Missing SS_Inf files on DGX1 -> Dealing with this now [Dealt with]


[Inferences]                                                 **[</data/ProbMaps_Inferences_Final>]**
- Physics segmentations: [BS1: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Physics_MPRAGE_3D> [LOCAL]
                          BS3: 50%    </raid/pedro/Experiments/BS3_ProbMaps_Physics_MPRAGE_3D> [DGX1]
                                      *Premature shutdown: Resuming inference for last 121 images: Had to clear space 28.06.19*
                         [SS_BS3: 100%]   </raid/pedro/Experiments/SS_ProbMaps_Physics_MPRAGE_3D> [DGX1]  <TRANSFERRED>

- Baseline segmentations: [BS1: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Baseline_MPRAGE_3D> [LOCAL]
                          [BS3: 100%]   </home/pedro/Project/SegmentationNetworkExperiments/ProbMaps_Baseline_MPRAGE_3D> [LOCAL]
                                        *Same folder as BS1: Seemed unnecessary to make a new folder for such a small change*
                          [SS_BS3: 100%]   </raid/pedro/Experiments/SS_ProbMaps_Baseline_MPRAGE_3D> [DGX1]  <TRANSFERRED>


# Meeting Notes:
Softmax done automatically when using a segmentation loss
No need to do softmax -> Argmax since RELATIVE values are the same

# Post-workshop cleanup:
- Assemble scattered notes + add sufficient detail for posterity

> MPRAGE Experiments
- Had to run multiple experiments with different hyperparameters to finally arrive at desired result:
  - Skull-stripped
  - (96 x 96 x 96) input window size
  - Batch Size = 3
  - No batch normalisation
  - "Dice_Dense" loss
  **NOTE** Found that loss was hard coded to "DenseDice", so ideal experiment was run with this loss despite being marked as "DenseDicePlusDenseXEnt"
  - (-0.2, 0.2) Dice range
  - Inference 160 OR 96

> SPGR Experiments
- Ran multiple experiments, minor improvements (as before, BUT never as good as 2D)
/home/pedro/NiftyNet-5/net_run.py inference -c /raid/pedro/Experiments/BS3_ProbMaps_Physics_MPRAGE_3D/Inference/Inference.ini

Method comparison:
SS_BS3 > BS1 >~ BS3
       > Dense_XEnt > BS1_LargeWindow
       > 96_inf (decent at WM)

DDXE seems VERY good, but realised that loss is hard coded to Dice_Dense: Coincidence then???
SPGR physics [Local], new blur (With DDXE) [DGX1], redo of MPRAGE physics (DDXE) [DGX1] running *Yet to try to look at tensorboard*
Initial blur experiments wonky: Had mistake with bias (-0.5, 05) INSTEAD of (-0.5, **0.5**) *Bias destroys image in first case*

Blur experiments: ProbMaps_Physics_All_white_LessBias_Lessnoise_blur
DGX1: 2 (Baseline SPGR), 4 (Physics SPGR divided PhysicsFlow10), 6 (Physics SPGR Attempt2 PhysicsFlow5), 7 (Inference SABRE) *All ProbMaps*
Local: SPGR Physics PF10

SPGR things (from months ago): /cluster/project0/possumBrainSims/GIF/SPGR_Inf_GIF/redos/Segmentations

SPGR experiments: /data/SPGR/UNetExperiments/ProbMaps_Inferences

SABRE GIF directory! </cluster/project0/possumBrainSims/SegTests/Data/GIF_SABRE_DoubleMPRAGE/>

## SASHIMI
- Major weaknesses identified + difficulty to address:
                               <!-- * GMM (GIF)for ground truth not seen as suitable, not similar enough to "true" human-like segmentations *Medium* -->
                               <!-- * Lacking in evaluation metrics: Why just CoV? Dice, Volume difference, MSD, HD <Hard> -->
                               <!-- * Lacking in references <Hard> -->
                               <!-- * Lacking in simulation method description, only described as "simple and robust" [Easy] -->
                               <!-- * Lacking description of SABRE dataset  [Easy] -->
                               <!-- * Don't justify simulation parameters  *Medium* -->
                               <!-- * Figure description lacking  [Easy] -->
                               <!-- * Lacking limitation discussion  *Medium* -->

# More comprehensive list of shortcomings and how they were addressed
* Lacking citations in the introduction: Added citation addressing acquisition variability
                                         *Volumetric analysis from a harmonized multisite brain mri study of a single subject with multiple sclerosis*
                                         Added citation addressing how variability can affect segmentation
                                         *Impact of acquisition protocols and processing streams on tissue segmentation of t1weighted mr images.*
* MPM acquisition + derivation lacking: Added citation addressing MPM protocol acquisition
                                        *Increased SNR and reduced distortions by averaging multiple gradient echo signals in 3D FLASH imaging of the human brain at 3T*
* Explaining parameter choice: Added citation addressing this
                                MPRAGE: *Optimizing the magnetization-prepared rapid gradient-echo (MP-RAGE) sequence*
                                SPGR: *Jog's paper*
* Contour maps lacking explanation: Gave more detailed description in results section
* Results description lacking: See above
  * Also no mention of segmentation consistency: Included table detailing DICE scores against PGS + p-value (Wilcoxon signed rank test)
* No discussion of limitations: Inclusion in final section of paper
* Aim of work not consistent: Re-worded offending passages (See MICCAI reviews for more detail on these)
* Lacking description of simulation model: Cited JOG's paper, extended description briefly

> Other changes (due to change in method)
* Network related: HRN -> UNet
                   2D -> 3D
                   Plots


# Log: 18.07.19 + 19.08.19
Aims: Cleanup SASHIMI notes + logs
      Analyse SABRE experiment (Due to awful contour plots)
      Look into non-SS networks

**[SS ProbMaps SABRE]**
For the "Inference" subjects (Used to construct contours) paired images are SIGNIFICANTLY different
  i.e. MPv2_[SUB_ID] vs MPv3_[SUB_ID] have significantly differing appearances
  Likely due to the fact that skull-stripping was done with HDBET instead of relying on GIF (As was done with the 12 "Train" subjects)

Re-did skull-stripping with GIF-based masks </data/Inferences_Final/GIF_SABRE/TIVs/Smoothed_masks/>
  Saved to (locally):      </data/SABRE/SABRE_res/SS_Multiple_Realisations_BC_Tots_${data_part}/SS_${data_part}/SS_${f}>
  and transferred to DGX1: </raid/pedro/Data/SS_Multiple_Realisations_BC_Tots_Inf>

Re-running inference:
/home/pedro/NiftyNet-5/net_run.py inference -c /raid/pedro/Experiments/Inference_ProbMaps_SABRE_3D_UNet_10_subjects/[...].ini
  *To be followed by re-doing contour plots*


**[Non-SS Experiments]**
Given "ugliness" and additional pre-processing required for skull-stripping method worth looking into NON-SS networks
  All probabilistic labels anyway, network should be able to generalise comparably
Running one such experiment: **NOTE** FCL == 10 (From SPGR experiments) NOT 5
/home/pedro/NiftyNet-5/net_run.py train -c /raid/pedro/Experiments/BS3_ProbMaps_Physics_MPRAGE_3D/[...].ini
  *To be followed by comparing to existing SS results*


Should use SAME mask for both images!





# Log: 22.07.19
Aims: - Investigate SABRE inference re-trial
      - Run inference on Non-SS BS3
      - Talk to Pritesh about student co-supervision (19-24 August)

**[SS ProbMaps SABRE]**
- Downloaded inference files locally: </home/pedro/Project/SegmentationNetworkExperiments/DGX1/Corrected_Inference_ProbMaps_SABRE_3D_UNet_10_subjects>

- Due to volume of changes, made new file: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/
                                           MPRAGE_inference_validation_paper_MiddleStandardisation.py>
  - Made ideal parameter calculations more robust: *Previously had found these out manually*
                            - Re-formulated plots: *interp2d -> griddata*

- Made contour plots: * Look reasonable, results NOT as good as with 2D
                      * Still amounts to small percentual differences between WM + GM (In fact WM better than before)
                      * <Re-running> TRAINING with Non-SS training examples

**[Non-SS Experiments]**
- Ran inference on BS3, files downloaded locally: </home/pedro/Project/SegmentationNetworkExperiments/DGX1/BS3_ProbMaps_Physics_MPRAGE_3D>
- Plotted against baseline + Physics: **NOTE** Not particularly good
                                      WM improves, GM very similar to baseline
                                      <Re-running> with FCL == 5 instead of 10

**[in2scienceUK: Pritesh co-supervision]**
- Spoke with Pritesh, main points: Not full time supervision, can set them tasks + encourage to visit relevant exhibitions
                                   Computer science main interests
                                   Spreadsheet of ideas: https://docs.google.com/spreadsheets/d/1DvEkrJ_I8V7usgA5VrCltmG00UM5uxWZZKSFKSc1VCI/edit?ts=5d3595d6#gid=0





# Log: 23.07.19
Aims: - Investigate FCL == 5 Non-SS BS3 experiment
        - Run inference *96 + 160*                    **[COMPLETED]** [Check tomorrow!] <Tmux6>
      - Investigate SABRE re-training (Non-SS)        *[In Progress]* [Check tomorrow!] <Tmux5>
        - *Added rotation augmentation, 10 degrees all axes*
      - Consider training "Baseline" SABRE network  *[In Progress]* [Check tomorrow!] <Tmux7>
      - Consider looking into plotting the ideal points for "Inference" subjects on contour plots
      - Consider plotting "Dice" similarity plots (Instead of just looking at volumetric differences): Likely to take a LONG time
                                                                                                     : Have to compare each v2 image with EVERY v3 image
                                                                                                     : scales as n^2, n == Images per subject, per protocol
                                                                                                     > Consider reducing [121 (40 hours)-> 61 (10 hours)]


# Group meeting Notes
New training protocols: - Multi-architecture with PP at start AND end
                          *Two separate FCLs or just one?*
                        - Freezing "Main" network + training just physics branch(es)





# Log: 24.07.19
Aims: - Investigate BS3 FCL5 Inference (96 + 160)
      - Investigate SABRE re-training -> Run inference on all subjects
      - Investigate Baseline SABRE -> Run inference on all subjects

**[New Experiments]**
- Multi-UNet: /raid/pedro/Experiments/Multi_Test_Physics/DD *Dense Dice LF, ONE physics branch shared*
              /raid/pedro/Experiments/Multi_Test_Physics/ *DDDX, ONE physics branch shared*

- Staggered U-Net: Train ONLY Main branch (~2000 iterations)
                    Train ONLY physics branch (~1000 iterations?)
                      Train both (~500 iterations?)
                   Allows for task specific training regimes + for bespoke hyperparameter assignment (i.e.: Higher when only physics)
>                  /raid/pedro/Experiments/Probmaps_staggered_training *Dense Dice LF, 1e-4 (Main), 1e-2 (Physics)*

- SABRE Inference on ALL subjects: Non-SS training complete, appropriate to run inference
>                                  /raid/pedro/Experiments/Inference_ProbMaps_SABRE_3D_UNet_10_subjects/No_Mask_Inference *10 train subjects*
>                                  /raid/pedro/Experiments/Inference_ProbMaps_SABRE_3D_UNet_12_subjects_checker/No_Mask_Inference *12 inf subjects*

[Changes/ Other notes]
- Renamed unet_same.py layer: 'R1_FC' -> 'Final'
                              This was to allow for ease of gradient exclusion for staggered training


# Jorge Meeting: Notes
To mention: * Probabilistic U-Net implementation: How to get latent space of parameters?
            * Multi-UNet: One or two branches?
            * Fat suppression: Plans for the future?
            * Weird behaviour with SPGR: With activated physics branch network wouldn't learn

Short-term: Iron out kinks in methodology: Address poor SPGR performance: * Compartmentalise network learning: Main branch -> Physics -> Both
                                                                          * prelu instead of just (l)relu
                                                                          * Multi U-Net architecture associated improvements

                                           Address variability in total tissue volumes of base vs physics
                                            *Note that this isn't crucial since it falls into accuracy, but would be good to understand*





# Log: 25-26.07.19
Aims: - Investigate BS3 FCL5 Inference (96 + 160) *Didn't check on these yesterday*
      - Investigate SABRE re-training -> Run inference on all subjects
      - Investigate Baseline SABRE -> Run inference on all subjects

[Changes/ Notes]
Network-related: <Staggered> Shifted to using unet_same_5FC so can keep naming in layers constant in original
                 <InferenceSABRE> SABRE_10 just fine, downloaded files locally: </home/pedro/Project/SegmentationNetworkExperiments/DGX1/Non_SS_Inference_ProbMaps_SABRE_3D_UNet_10_subjects>
                  **NOTE** Had some errors with running SABRE_12 due to network layer naming mismatch *Re-run*

Multi-UNet: Running inference for DD + DDDX (96)

Staggered: Ran into errors when trying to train w/ physics after freezing main
           Apparently network REQUIRES that all parameters ALREADY be initialised when starting from a checkpoint
            Solution: Run FULL network for one iteration (So everything Adam related initialises)
                      Start from this checkpoint with ONLY main
                      -> Physics
                      -> Both





# Log: 29.07.19
Aims: - Run multiple main branch frozen version of Physics staggered U-Net
        - lr: 1e-4, 1e-3, 1e-2, 1e-1, 1e0
      - Carry out plots for new SABRE inferences
      - ConvNet tutorial for in2scienceUK students
      - SABRE inference with SAME masks *REALLY minimise variability arising from masks as much as possible*

[SABRE contour plots]
- Look good, but ideal points look odd: All along contour line (V. good) but not near the "ideal" region

Anti-transpose: [::-1, ::-1].T
  See also: https://stackoverflow.com/questions/44772451/what-is-the-best-way-to-perform-an-anti-transpose-in-python

[Changes/ Notes]
qstat: qstat | head -<N> (Place with other info at file header)
.


# Log: 19-23.07.19
> in2scienceUK week
- Ongoing summary...


> Research
[Staggered training]
Models taken from main branch only training (Up to 4k iters) -> Placed in each of the staggered training folders *1e0, ..., 1e4*
  Staggered training taken from there: 4000 -> 8000 iterations

Running inference on all of these </raid/pedro/Experiments/ProbMaps_staggered_training>





# Log: 27.08.19
Aims: - Investigate staggered training [Y]
      - Investigate altered staggered training [Y]
      - Look into multi-UNet training [Y]

[Staggered training]
Recap: Trained UNet only network for 4000 iterations  </raid/pedro/Experiments/ProbMaps_staggered_training>
       (1) Froze ALL of main network and unfroze physics branch: Trained for 4000 more iterations, using different learning rates  *1e0, ..., 1e-4*
       (2) Froze MOST of main network EXCLUDING final block of layers (between physics & output), using different learning rates   *1e-3, 1e-4*
      **NOTE** (2) is the CORRECT way to do this, otherwise final layer weights do not update in response to physics

> Current developments
Have trained (2), currently running inference *Consider 160 inferences*

Current rankings: Standard 160 inference DD network performs best >
                  Partial physics > Physics Only

[Multi-UNet]
Trained "joined" model where there is a SINGLE physics branch that feeds into multiple points in the network (Before + After downsampling)
  </raid/pedro/Experiments/Multi_Test_Physics/>            + </raid/pedro/Experiments/Multi_Test_Physics/DD>
**NOTE** This might not be the best approach: Might want to have a separate branch for each parameter passing
  </raid/pedro/Experiments/Multi_Test_Physics/Separate_DD> + </raid/pedro/Experiments/Multi_Test_Physics/Separate_DDDX>





# Log: 28.08.19
Aims: - Staggered training: Further work
      - Multi-UNet work continued

[Staggered training]
Given "decent" performance of 1e-3 staggered training, should run inference with 160 inference _WIP_
  *Can also consider using GS network as pre-trained network*


[Multi-UNet]
Separate_DDDX Inference: Not bad, rivals current GS -> 160 inference NOT as good as 96 **Actually better on average than GS, though bizarre curves**





# Log: 30.08.19
Aims: - Multi-UNet (continued)
      - Staggered training (continued)

[Multi-UNet]
160 vs 96 inference: 160 seems significantly better

Thought: For "Baseline" why not run inference on PHYSICS network with some "standard" TI value to be passed for all images
         If parameter is always the same then in theory it's as if the network isn't accounting for physics at all
          Running these experiments under </raid/pedro/Experiments/Multi_Test_Physics/Separate_DDDX/Bad_Physics_Inference_Test(_0600s/2000s)>

> Outcome
As expected, no visible invariance observed (Looks similar to standard Baselines)
Outlier: Running inference with TI=2000ms (Unseen value): Network GM invariance significantly higher than 600 or 900 TI for some reason?
          Perhaps need to account for OVERALL CoV instead of tissue specific





# Log: 9 - 12.09.19
> CDT Edinburgh Summer School





# Log: 06.09.19
Aims: - Multi UNet (DD): Compare inference
      - Staggered training: Reminder of current status + continuation

[Multi_UNet]
Current GS order: Separate DDDX 96 > Separate DDDX 160 > Standard DDDX 160 *previous GS* > Separate DD 160 > Separate DD 96
                  **NOTE**: Attempt with SPGR (Since results were lacking before)

DD comparison: Significantly worse than DDDX, therefore opt for the latter

[Multi-UNet SPGR]
SPGR experiments: Running various training instances to evaluate variance </raid/pedro/Experiments/Multi_Test_Physics_SPGR>
                  *Default, Run2, Run3, Run4*




# Log: 09.09.19 @@
[Multi-UNet SPGR]
Might have some issues relating to SPGR inference: Running inference on **21** instead of **5**

Preliminary analysis of inference when compared to GIF and current GS: **LACKLUSTER**
                                                                       Very large variability for some reason
                                                                       Could be due to halting network prior to actual finishing of training

[Fat-water shift: Recap]
Chemical shift arises due to differing resonant frequencies of protons in water vs fat    https://mriquestions.com/f-w-chemical-shift.html
  In water molecule arrangement, _Oxygen >> Hydrogen_ in terms of <ELECTRONEGATIVITY>     https://en.wikipedia.org/wiki/Electronegativity
    Causes Oxygen to pull covalent electrons <CLOSER> to it
    => <DESHIELDING> effect, exposing hydrogen protons => Experience a larger local magnetic field
      Frequency difference == Larmor Frequency x 3.5ppm


**Things to do**  ##THINGSTODO
>Implement ability to dynamically set size of FCL in physics networks                                    [DONE]
>Implement generalisability of volume plots to many more experiments (Not just Physics, Baseline, GIF)   [DONE]
Devise comprehensive training protocol: 1. Staggered training should be default, 3 runs                   _WIP_
                                        2. DDDX
                                        3. DD
                                        4. Separate DDDX
                                        5. Separate DD

**[China Visa]**
Online application form                                         [Completed]
Appointment: Booked for 9:30, 23/09/19                          [Completed]
Documents needed: - Printed application form                       <No>
                    - *Don't forget to sign as is appropriate*
                  - Physical photo copy                            <No>
                  - Flight confirmation (In and out of China)      <No>
                  - Accommodation for entirety of stay             <No>
                  - Photocopy of passport photo page               <No>





# Log: 10.09.19
Aims: - Experiment monitoring
      - Inference comparisons (Multi-UNet DDDX Runs on SPGR)

[Current Experiments]
Multiple runs of Separate DDDX Multi-UNet for SPGR: </raid/pedro/Experiments/Multi_Test_Physics_SPGR/LongerFC_Run>      *[In Progress]* <Tmux7>
                                                    </raid/pedro/Experiments/Multi_Test_Physics_SPGR/LongerFC_Run_2>    *[In Progress]* <Tmux6>

Separate DDDX Multi-UNet w/ Longer FCL for MPRAGE: </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run>            *[In Progress]* <Tmux5>
                                                   </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run_2>          *[In Progress]* <Tmux8>


> Python extras
isinstance(VAR, condition) *True or False*
assert CONDITION [, OPTIONAL_STATEMENT]
all(CONDITION) *Returns True only if True for ALL elements*





# Log: 11.09.19
Aims: - Code cleanup
      - DGX1 monitoring

[Post-processing code cleanup]
- Plotting code is clunky and bloated
- Hard to expand, most things are hard-coded to three experiments *Physics, Baseline, GIF*
  - To do: <Generalise> Make adding new directories to analyse easy
           <Slim-down>  Transition to more functional programming

*Start on: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/MPRAGE_inference_validation_paper_MiddleStandardisation.py>*   **[DONE]**
           Tests: Basic (Three vanilla directories)    [PASSED]
                  Medium (4+ directories)              [PASSED]
                  Advanced (4+ directories, TIV, CSF)  [PASSED]
To follow: SPGR, MPRAGE (Non-middle standardisation)

[SABRE Experiments]
Trained SS + Non-SS networks using new multi-UNet architecture </raid/.../ProbMaps_Physics_All_white_LessBias_Lessnoise_blur_Multi(/SS)>
  Now running inference: </raid/pedro/Experiments/Inference_ProbMaps_SABRE_3D_UNet_10_subjects_Multi/> <Tmux4>      _WIP_
                         </raid/pedro/Experiments/Inference_ProbMaps_SABRE_3D_UNet_10_subjects_Multi/SS> <Tmux3>    _WIP_


[Other]
Interesting blog post about knowledge distillation: https://towardsdatascience.com/knowledge-distillation-and-the-concept-of-dark-knowledge-8b7aed8014ac
  <Gist>: * Transfer generalisations of complex model to lighter model
            **NOT** the same as _Transfer Learning_: TL involves transferring pre-trained weights to other network of SAME architecture
          * KD involves transferring GENERALISATIONS learnt by complex network to a smaller version
          * Concept of SoftMax temperature: If alter SoftMax with T (Temperature) can "soften" the output probabilities
            * Effect: Gain insight into _Dark knowledge_ of the network, i.e.: info about classes most similar to predicted class
                      Can use these concepts for training the student network:

             ([High Temp Teacher Output] + [OHE labels]]   <DistillLossFn>   [High Temp Student Output] + [Low Temp Student Output])






# Log: 12.09.19
Aims: - Code cleanup (continued)
      - SABRE, Longer FC tests check

[Post-processing code cleanup]
- MC code cleanup  *Yet to be tested*
- All tests passed for MPRAGE minus DICE

[SABRE Experiments]
- Running inference on 10 + 12_checker: Non-SS & SS

**NOTE** Made mistake in network selection!
          **Using unet_physics_same_multi INSTEAD of unet_physics_same_multi_separate!**

<Separate_DDDX>       <WRONG>
<Separate_DD>         [CORRECT]
<LongerFC_Runs>       <WRONG>
<SABRE_Experiments>   <WRONG>

- Separate_DDDX became new GS, therefore NON-Separate seems to be best approach for now
  - BUT Investigate Separate properly!


# Meeting Notes:
To mention: - Currently running multiple instances of each network
            - Also testing out longer FCL hypothesis

Questions/ discussion: - GIFNet: Useful to do parcellations?
                       - SPGR: Maybe try toy example? Create simplified MPMs (spherical?)

Fat shift: No success, best bet is probably POSSUM

> Feedback:
SPGR Toy example: * No go, no reason to think that the problem isn't tenable
                  * Better to go about it freezing route, more params == more chaotic gradients == smaller momentum == more negligible updates
                    **NOTE** Ask Carole about this!

Distillation: * Feature comparison prior to (1 x 1 x 1) convolution
              * Stratify batches such that each batch contains same subject, same patch, DIFFERENT realisations
                * Therefore encouraging network to push gradients more towards physics invariance
                * Have an L2 between features: Because label is the same features should be similar as well
#                 Need to figure out implementation details!

Graphs: Need to do best to account for kinks in graph, not "natural" therefore need to justify it
        I.e.: More complex FCL (More neurons, more connections, etc.)
        Also best to maybe avoid non-differentiable activations for this reason
          *Since physics being modelled will be much smoother than MR image -> label function*
          See: selu: tf.nn.selu
               elu:  tf.nn.elu
               Relevant blog posts (e.g.: comparisons): https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9 (SELU: Quant. Comp.)
                                                        https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9 (SELU: Overview)

**[SUMMARY]**
> Carole + network freezing + development
> Stratified batch development: 1. Ensuring each batch contains ONLY images from ONE subject
                                2. Ensuring the SAME patch is always picked within a batch
                                3. Coming up with best feature comparison method
> FCL development: More layers, more neurons, new activation functions (SELU > ELU)





# Log: 13.09.19
Aims: - Investigate inferences
      - Begin developing startified batch

[SABRE]
Non-SS results pretty poor, completely disregard unless planning to SS in post *Maybe worth doing so a more direct comparison with SS can be done?*
SS results exhibit obvious erroneous label classification outside of brain for some reason
  - Best to correct in post
  Created a modified version of <SABRE_inf_organiser.sh>, <SABRE_output_inf_organiser.sh>:

    1. Loops through files in current directory
    2. Depending on length of filename extracts ID and finds corresponding Mask in mask directory </data/Inferences_Final/GIF_SABRE/TIVs/Smoothed_masks/>
       and multiplies mask with file *(Stupidly filenames are not all same length due to differing ID lengths)*
    3. Outputs file to desired directory

Carried this out on both SABRE output directories:
</home/pedro/Project/SegmentationNetworkExperiments/DGX1/Inference_ProbMaps_SABRE_3D_UNet_10_subjects_Multi/SS/Post_SS_Files>
</home/pedro/Project/SegmentationNetworkExperiments/DGX1/Inference_ProbMaps_SABRE_3D_UNet_12_subjects_checker_Multi/SS/Post_SS_Files>
Run bridging on both _WIP_

[Other]
Made modification to <mass_renamer_ini.sh> to NOT search in subdirectories by adding [-maxdepth 1] option to find command
  Was running into issues when subdirectories had .ini files themselves

[SELU]
Running experiments with SELU activation functions in physics branch (Instead of leaky-relu)
</raid/pedro/Experiments/Multi_Test_Physics/Joined_DDDX_selu>

[LongerFC]
MPRAGE: </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run>
SPGR: </raid/pedro/Experiments/Multi_Test_Physics_SPGR/LongerFC_Run(_2)>





# Log: 17.09.19
Aims: - Last week Experiments check
      - Stratification implementation

**[Experiments check]**

SELU: [MPRAGE]     *Inference*
LongerFC: [MPRAGE] *Inference*
          [SPGR_1] *Inference*
          [SPGR_2] *Inference*

**[Stratification implementation]**
Main features: 1. Each batch contains only images from ONE subject
               2. Each batch features only a SINGLE patch location

> Patches
Coordinate calculations: _spatial_coordinates_generator produces array length 7:
                                    <Array>[PatchID, x_min, y_min, z_min, x_max, y_max, z_max]
 Was able to check this by printing out coordinate arrays and checking that *x/y/z_max = x/y/z_min + x/y/ patch length*


shuffle queue off


> Re-write properly tomorrow!!
Summary: 1. Remove hard coding of queue length (In image_window_dataset, self.queue_length variable for searching purposes)  ##HARDQUEUE
         2. Remove shuffle from: image_reader, uniform_sampler
         3. Make sure queue_length == batch_size, ensures no randomisation (Only within batches which is fine, i.e. [1,2,3] == [3,1,2])
         4. Patch creation moderated by seed obtained from image_id: idx // batch_size: Ensures that within each batch there will be the same (unique) seed
           4a. Use floor division to obtain idx

Checks to be covered: 1. Labels should be identical  *Implemented stratification_label_check for this*
                        1a. Calculation:              [LabelA - 0.5(LabelB + LabelC)] == 0
                      2. Images should NOT be identical  *Implemented stratification_image_check for this*
                        1a. Calculation:              [ImageA - 0.5(ImageB + ImageC)] =/= 0
                      3. Two points above should generalise to validation  *Ensure val. set comprised of full subjects and not split slices*
                      4. By readjusting volumes per subject to be // 3, all points above should ALWAYS hold





# Log: 18.09.19
Aims: - Stratification implementation (continued)


[Quick summary]
SELU: Seems good, check plots (Still open atm)
Stratification: Implemented feature loss: 3 combos within batch (L2), then average
                Tried a few loss combinations: root(FG) [BAD], root(F) + root(G) [POOR], F + G [SUBPAR]
                  Seems like feature loss takes over and sets everything to zero
                    Best to only turn it on after network has learned segmentation somewhat
                    Also had issues with spikes (i.e.: Non-stratified batches: *UNSOLVED*)
                Followed in footsteps of lr scheduler, keep looking into this ()
#                iteration_message.data_feed_dict[self.learning_rate] = self.current_lr


**[Stratification implementation]**
Fill in later date...





# Log: 19.09.19

Dan talk: A General and Adaptive Loss Function (https://arxiv.org/pdf/1701.03077.pdf)
          Tangential discussion: Perlin/ Simplex noise, robust loss function paper
Stratification: Implemented ability to change loss function during training (DDDX -> DDDX + feature) ~ LR scheduler (data dict takes bools)
                Tested on Baseline examples (40 good, i.e.: stratified) -> Baseline folder (Limited dataset)
                  Also started testing on full dataset (>600 activation): FullBaseline
                Had to make sure that vols/subject // batch_size (exclude 1.200)
                Had problems every 60 iterations (BS 2): Found out there were 121 vols for validation instead of 120, moved everything up by one
                  therefore every 60 iterations (120 vols processed) there would be a spike in feature loss because vols came from different subs (even though seed was correct)
                Had to move to BS 2 (from 3) due to OOM issues: Investigate if problem persists on DGX1
                General cleanup of code -> Began generalising to Physics (Only Physics Sampler)

                **NOTE** Probably good idea to add regulatory (SegLoss/FeatureLoss) term to prevent FL from collapsing to very small numbers
                         See run 4 in FullBaseline: At iteration 1120 FL drops significantly and SL jumps accordingly: Stuck in bad solution space
                          BUT can see that that corresponds to a large jump in SL/FL graph: ~2 -> ~90: Suggests this could be a good addition to loss





# Log: 20.09.19
Aims: Implement stratification for Physics network + Sampler
        Modify loss function to include regulatory SL/FL term (Need to pick an alpha)
        Consider SSIM term to compare (real) images within the same batch?
      Deploy changes on DGX1 and test further there
      (Clean up notes from past 3 days)





# Log: 23.09.19
Aims: - Examine over-weekend experiments of stratification with regulatory loss term
      - Examine SPGR inference for LongerFC experiments


[Stratification]
L2 loss: CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training
                  https://arxiv.org/pdf/1703.10155.pdf
         Potentially better alternative for feature matching:
>                                                         0.5 * || F1 - F2 ||^2_2    (Page 4)   *This is just an L2 over the features*
          Syntax: https://stats.stackexchange.com/questions/181620/what-is-the-meaning-of-super-script-2-subscript-2-within-the-context-of-norms
> Loss is identical to standard L2 Loss... Good I guess?


Deployed changes to DGX1, running experiments under:  </raid/pedro/Experiments/Stratification_Tests/Baseline>
  image_window_dataset (forced queue size logic), image_reader (shuffling), application, network (Exposing features to application)  ##QueueStrat
Can run with batch size 3 now locally! Problem related to having an instance of tensorflow open in a terminal instance of Python (Had to just close it)

Patient shuffling: Don't forget to implement!


[Other]
Useful LaTeX equation creator tool: https://www.codecogs.com/latex/eqneditor.php
If a variable is a list of instances of a class and you want to call a class method for each:
>                                          OUTPUT = [EXAMPLE.METHOD() FOR EXAMPLE IN ORIGINAL_OUTPUT]





# Log: 24.09.19
Aims: - Think aboutu shuffling implementation
      - Carry out non-feature loss experiments: Performance exceeded?

[Shuffling stratified batches]
Due to current implementation, subject images are passed to network consecutively *i.e.: ALL of subject 1 images train -> ALL of subject 2 images etc*
NOT ideal, likely training is hindered due to gradient being overly contingent on individual subject anatomies
Solution: Shuffle images in groups of batch_size: This likely requires a naming trick of sorts
          Idea one: Add an incremental prefix (Post string matching) number: 1. Create lists of each of the subject images
                                                                             2. Randomly pick one subject
                                                                             3. Pop first three entries, add prefix, save with new name
                                                                               3a. Note that labels and params need to be adjusted accordingly, too
                                                                             4. Loop until all images processed
                                                                             5. Likely need Python script for this

> stratified_shuffler.py  ##SHUFFLER
1. Sorts directories according to subject ID
2. Create list containing all multiples of batch_size up to (#imgs / batch_size): Indexing to be sampled from this list
3. Sample randomly from batch_ID_list, index images/labels/params, add images to new "shuffled" list
  3a. Remove transferred images from original lists, remove final entry in batch_ID_list (Since indexing doesn't go that high anymore)
4. Rename files



# Log: 25.09.19
Aims: Finish shuffling implementation

[Shuffling]
> stratified_shuffler.py
Found that renamed_images =/= renamed_labels
  This was due to the fact that even though images/ labels lists were ordered according to subject ID, the TI ordering is **RANDOM**
  Solution: Easier and more computationally efficient to make renamed image list and create an exact copy with an altered prefix (for labels)

Directories: Images: </raid/pedro/Data/MPRAGE_LowTD_121_Train/SS_Train/Test_environment/renamed>
             Labels: </raid/pedro/Labels/Physics_Gold_Standard/Labels/Test_environment/renamed_labels>
             Experiments: </raid/pedro/Experiments/Stratified_Experiments/Baseline/Shuffled>

>Issues
While _stratification_label_check_ is MOSTLY zero, there are some exceptions
  Can't find reason for this since images that are supposedly in same batch exhibit same patches
  HOWEVER, examples exist where intra-batch images have different patches, suggesting seed is not being reset properly
    **Potential solution**: * Problem has to lie with seed, therefore try to instance seed as "deeply" as possible
                            * Since patches HAVE to be generated for each image by placing seeding in that section of code should ensure expected behaviour
                              * i.e.: <SEED>: [Layer_op] TO [rand_spatial_coordinates]
                                * *The latter has to be converted from function to method so self variables can be passed to it*

[Other]
Selectively carry out pattern replacement in Vim: https://stackoverflow.com/questions/58096643/replace-string-only-in-matched-search-in-vim/
  Use [:g/<SEARCH/ s/<PATTERN>/<REPLACEMENT>/]



## STRATIFICATIONNOTES
Potential shuffling resolution: Create large array containing MANY randomised array locations
                                Do this by calling np.random.randint([SAMPLER CODE]) for N outputs
                                  Then index in same way that seed was being generated (Image_ID // batch_size)
                                  NOTE: Too deterministic, should find a way of indexing according to iteration in addition to subject ID

max_coords in sampler: For EACH dimension do the following: [random_int(0, max(image_dimN - win_dimN + 1, 1), 1)]: static for each dim.

Application: self.data_param -> Sampler (as window_sizes)
                             -> ImageWindowDataset (as window_sizes)
                             -> ImageWindow.from_data_reader_properties(as window_sizes)
                             -> ImageWindow.set_spatial_shape(as spatial_window)
                             -> **HERE**           vars(self.data_param['MPRAGE'])['spatial_window_size']    *MPRAGE is an example of a name*

What about image size? -> No way to access besides passing through an image reader
                          Best way to approach this is to pass it as another variable in the config. file
                          **[DONE]** <seed_image_size> parameter under Segmentation section

Issues: SOLVED- Had to deepcopy array, otherwise was always adding "half_win" and increasing patch locations s.t. > image





# Log: 01.10.19
Aims: Stratification continued...
      SPGR results analysed (Longer FC)
      Other minor pending ToDos (Cluster access, Fixup details, SPGR code cleanup)

> Corrections
stratified_shuffler.py: Correct s.t. one subject (At random) is selected to be validation
                        Place subject at "the end" (in terms of training numbers)
                        *Actually not necessary as long as validation batches treated the same as training batches*

[Stratification]
- Carried out more comprehensive tests for seeded array method of patch homogeneity: **Success!** *100% intra-batch consistency over 1000s iterations*
  - Works for ordered + unordered
- Need to focus on unordered method now

> LR scheduler parameters
t_mul: 1.25
m_mul: 0.85
alpha: 1e-5
first_decay_steps: 2590

SPGR analysis: Great! Plotting TIVs
               Running another Baseline just to double check
               **CLEAN-UP SPGR CODE** Added TIV option for more in depth consistency check (Leaving after this)
Stratification: Good, seeded array work seems to perform well: stratification_label_check is ALWAYS zero now (Fluctuations before)
                Running for unordered batches
                Idea to multi-scale probably not feasible due to static nature of graphs
                Added growing multiplicative epsilon term to total loss and made sure that feature loss is included from get go (Easy implementation)
                As segmentation becomes better the importance of homogenising features should grow
                Changed UniformSampler [DGX1] + segmentation_application [DGX1] to NOT have stratified behaviour *Reserved for Physics Experiments for now*
UCL Cluster: Regained access (Use comic for standard login, beaker for GPUs)
Fixup + Cosine LR scheduler: Have code + params, need to just spend time implementing
Other: [netdel.sh <PORT>] Allows freeing up of open ports for non-local tensorboard casting





# Log: 02.10.19
Aims: - Stratification implementation tuning
      - Inference on stratification networks

Stratification implementation tuning: Seeded array works well BUT due to current implementation patch selection is deterministic for Epoch > 1
(All in sampler_uniform_v3 of course)   This is due to seeded array indexing == seedID == subject_ID
                                        Therefore when subjects loop, so do patches for those specific images
                                      **SOLUTION**: - Once patch has been used for a batch, randomise the previous one
                                                    - OR, once epoch has finished, randomise whole array (This may be far more efficient)
                                                      - Works! </raid/pedro/Experiments/Stratification_Tests/Physics/Tests> *Randomises on image ID 2000*

> Ordered worse performer:
Could be due to FCL being TOO large OR pseudo-randomisation instead of full-randomisation, will investigate both

> Contour plots
Working on it, making decent progress (plotting for single experiment, single tissue: Phys_GM)





# Log: 03.10.19 + 04.10.19
Aims: - Stratification re-evaluation
      - Unordered/ ordered redo

> SPGR plots





# Log: 07.10.19
Aims: - Advanced ML course meeting
      - Group meeting (Ron)
      - SPGR contours (continued)


Due to random nature of sampling, re-doing inference with 11 x 11 x 3 (TR x FA x TE)
  can keep one parameter constant at a time + reasonable spanning of parameter space





# Log: 08.10.19 - 10.10.19
Aims: ...

Uploaded FullSampling SPGRs to DGX1  </raid/pedro/Data/SS_SPGR_FullSampling> </data/Simulations/FullSampling_SPGR>
Seg Faults under new Dockerfile: 1. Adjusted logging, no results
                                 2. Pulled latest NiftyNet changes (Changed origin to https://NifTK/NiftyNet)
                                   2a. Resolved conflicts: Negligible, except for segmentation params added for mixup
                                 3. No results, went back to older container
                                 4. Had errors, reverted changes made in (1.), errors persisted
                                 5. with_bn = False -> feature_normalization = None in required networks
                                 6. Completely reverted windows_aggregator_grid.py: [if windows] logic seems to break/ not be fulfilled
                                   https://github.com/NifTK/NiftyNet/commit/fc87e0ac54ad799639478ea6d147302b409648a0 (Under decode_batch)

Additonals (From memory): Need to segment all SPGR volumes using GIF to have apt comparisons
                          Uploaded all FullSampling SPGRs to UCL cluster in preparation of this:





# Log: 11.10.19 - 30.10.19
> MICCAI (13 - 17) + China trip (18 - 30)





# Log: 31.10.19
Aims: - Work re-adjustment
      - Check on SPGR segmentations on cluster
      - Check on other pending/ completed jobs on DGX1

**[FullSampling SPGR GIF segmentations]**
**ERROR** Ran out of storage part of the way through: </cluster/project0/possumBrainSims/>
            Only had 1TB allowance, completely filled during segmentation process: Completed 861/2178 segmentations (~40%)
            Segmentation files WERE created, but all empty

@Solution: 1. Ask for additional space, 1TB, from storage-request@cs.ucl.ac.uk: [Granted speedily]

           2. Organise directory as normal                                      [extract -> GIF_organiser.sh*]

           3. Find and move those files whose segmentation was NOT completed:   [find . -type f -size 0 -exec mv {} DIRECTORY/ \;]
              https://stackoverflow.com/questions/20014201/how-to-move-all-files-with-fewer-than-5-bytes-in-bash *Command explained in full*
             3a. Files moved to unique folder </cluster/project0/possumBrainSims/GIF/FullSampling_SPGRs/Zeros>
             3b. Created <redos.sh>: Loops through /Zeros/, moves files with corresponding name to /Redos/ from folder /OGs/
             3c. Run GIF script in </cluster/project0/possumBrainSims/GIF/FullSampling_SPGRs/OGs/Redos>

           4. ALSO have files that were not addressed at all *Probably never had a chance to run on cluster before storage issue*
             4a. Identify those missing files by looking at missing segmentation files (2160/2178 exist)
             4b. Move all empty folders from <GIF_organiser.sh> command to new directory
             4c. Create a directory of "names" (i.e. empty files with all names of all images)
|                  for direc in $PWD/*; do
|                    touch $PWD/all_files/$(basename $direc).nii.gz
|                  done

             4d. Find missing files by running diff:
              https://stackoverflow.com/questions/16787916/find-the-files-existing-in-one-directory-but-not-in-the-other
|                  diff -r /cluster/.../empty_folders/all_files
|                          /cluster/.../Segmentations/ |
|                       grep /cluster/.../empty_folders/all_files | awk '{print $4}' > difference.txt  __Saves files in difference.txt__

            4e. Loop through lines in difference.txt, move files from /OGs/ -> /Exclusions/ & run GIF
                </cluster/project0/possumBrainSims/GIF/FullSampling_SPGRs/OGs/Exclusions>
                   https://www.cyberciti.biz/faq/unix-howto-read-line-by-line-from-file/
|                  while read -r line; do COMMAND; done < input.file

- Now running ALL missing jobs. ETA: *Unknown*

**[FullSampling SPGR GIF segmentations]**
- SPGR plots: Physics + Baseline (11 x 11 x 3)
  - In progress, running code as a reminder of point left off

[Others]
- <Files>-type command for counting directories: [find . -type d | wc -l] *d instead of f, essentially*
- Fixed netdel.sh <LOCAL> such that killing of connection works regardless of PID length
  - Employed regex within grep to find only digits on matching line: [grep -oP '\d*'] *o: Only matching, P: PERL style regex*
- `<`: Used to redirect file contents, can write for loop and at end direct file as input by using `<`
- Contours information (irregular plots): https://stackoverflow.com/questions/27004422/contour-imshow-plot-for-irregular-x-y-z-data
- Zero centering of colorbar: https://stackoverflow.com/questions/25500541/matplotlib-bwr-colormap-always-centered-on-zero





# Log: 01.11.19
Aims: - Look into CBD dMRI dataset (Multi-dimensional Diffusion-Relaxometry MUDI 2019- MICCAI 2019)
        - Research means of dMRI simulation
      - ...

**[CBD data + related simulations]**
Diffusion MRI signal papers: https://mriquestions.com/uploads/3/4/5/7/34572113/e_version.pdf (Signal equation derivation)
                              https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6429954/pdf/nihms-1010239.pdf (Easier to read version)
                             https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0149778&type=printable

Gist: dMRI signal is a function of unattenuated signal modulated by an exponentially decaying function with b (Diffusion coefficient)
        S = S0 exp(-b*ADC), where ADC =  ln(S0/ S(b))
      For CBD data can therefore (presumably) take images of constant b to ignore b component
      Other info: * Columns: b-vector (x, y, z), b-value, TI, TE
                  * TR is constant: 7500ms (Refer to email)
                  * Ask about ADC? (According to paper should be ~ 0.001 m2 s-1)/ IE ~ 2 (Inversion efficiency)
      Seems like images are IR-dMRI: Use IR signal equation (See: https://cni.stanford.edu/wiki/MR_Signal_Equations)
                                      More complete equation including TE: http://mriquestions.com/choice-of-ir-parameters.html
                                      [S = k * PD * (1 – 2 * exp(-TI / T1) + exp(-TR / T1)) * exp(-TE/T2)]
                                     CBD paper: https://kclpure.kcl.ac.uk/portal/files/102443829/s41598_018_33463_2.pdf
                                     CBD resources (data, citations, etc.): http://www.developingbrain.co.uk/data/
                                      http://cmic.cs.ucl.ac.uk/cdmri/challenge.html

Reading new parameters file: 1344 per subject

token = open('parameters_new.txt')
linestoken = token.readlines()

resulttoken = []
for x in linestoken:
    resulttoken.append(x.strip(' ').strip('\n'))

final = []
for x in resulttoken:
    final.append(x.split(' '))

final = np.array(final)

spgr_copier.sh /data/Simulations/All_subjects_SPGR_RandomSampling /data/Labels/RandomSampling_templates
label_copier.sh /data/Labels/RandomSampling_templates ProbMaps_RandomSampling_Labels
maker.sh ProbMaps_RandomSampling_Labels





# Log: 04.11.19
Aims: - Full Sampling Training setup (LOCAL)
        - Create label templates (spgr_copier.sh)
        - Create labels (label_copier.sh)
        - Skull strip images (big_masker.sh / big_spgr_masker.sh)
        - Create training directory (</home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR>)

> Potential fixes:
- No need to actually copy files for label templates, all that's ncessary is to create file **NAMES** in required subdirectory structure
  - Can use *touch* command for this purpose, creates empty file of desired name
  - Will save LOTS of time

[Others]
Want to delete file even if they don't exist (which would normally raise an error): 1. -f option: Force regardless of permissions
                                                                                    2. Use OR: rm FILE || true *If fails returns true and moves on*

SPGR training images: </data/Simulations/FullSampling_SPGR_Train>
SPGR training labels: </data/Labels/FullSampling_train/Train>

- <FSLSTATS> Check image statistics: * Min & Max: fslstats <IMAGE> -R *Useful to identify presence of infinites and/or NaNs*
              Correction use case:          [for f in Label_spgr_sub_*; do seg_maths $f -uthr 1.1 $f; done]
                *In theory should threshold at 1.0, but many points in image are JUST above 1.0, so -uthr 1.0 causes issues*

**[iSMORE paper for SIE seminar]** http://link.springer.com/content/pdf/10.1007%2F978-3-030-32778-1_14.pdf
Criteria for presentation: - 5 minutes max.
                           - Purpose of highlighting a contribution
                           - 2/3 slides
                           - Pros/ cons, applications, limitations, developments + extensions





# Log: 05.11.19
> Notes: Keywords Self super-resolution, deep networks, SMORE
- Motivation: * High-res images are always preferred in medical images
              * BUT usually only have HR for in-plane directions
                * LR in perp. planes
                * Cannot solve this by simple interpolation
                * This is when **SUPER-RESOLUTION** methods come in

*ASIDE* Super-resolution: * Requires LR/ HR paired training data
                          * HOWEVER, this paired data is often lacking: **SELF**-SR methods proposed to solve this
                            *No need for external training data*
                          * Method: 1. Degrade HR in-plane acquisitions
                                    2. Use generated LR images paired with OG HR images to train network
                                    3. Use trained network on perpendicular (LR) planes
                          **NOTE** In-plane slices are usually thick (averaged) HR slices *Suboptimal*
                                   Flawed methodology of applying 2D trained network on 3D data

- Purpose: * Iterative Super-resolution algorithm that outperforms current methods

- Methods: * Can be separated into 2D and 3D
           * _Standard 2D framework_: 1. Input image __g__ of resolution (a x a x b) [a: Good resolution, b: Bad resolution]
                                      2. Take HR in-plane xy (a x a): Blur using PSF (Mimic LR) -> (b x a)
                                      3. Train networks on this paired data: (b x a) -> (a x a)
                                      4. Apply this on off-plane zx slices (b x a) to obtain HR slices (a x a)
                                      5. Stack zx slices to get full HR volume: (a x a x a)

             _iSMORE framework_: 1. Take OUTPUT of standard 2D framework [f1] as starting point
                                 2. Take NEW (More) HR in-plane xy slices as starting point, repeat steps, apply on __g__ ==> [f2]
                                 3. Repeat steps 1 - 2 until convergence condition is met

           * _Standard 3D framework_: Largely Non-existant, reliant on 2D CNNs
                                      Consider: 1. Degrade your input __g__ (a x a x b) -> (b x a x b)
                                                2. Train network to learn this mapping
                                                3. **HOWEVER** mapping unsuitable for SR purposes: Training LR == (b x a x b)
                                                                                                   Actual LR == (a x a x b)

             _3D iSMORE framework_: 1. Run one iteration of 2D iSMORE -> (a x a x c) image [c ~ a]
                                    2. Degrade image -> (b x a x c)
                                    3. Train network to learn this mapping: (b x a x c) -> (a x a x c)
                                    4. Apply on __g__ ==> (c x a x c)

- Other details: * Loss function: L1 + Sobel filter loss
                 * 3D network: Wasteful to use (3 x 3 x 3) convolutions since only ONE dimension is LR
                               More efficient to employ (3 x 3 x 1) or (3 x 1 x 3) kernels (Assuming first dimension is LR)
                 * Consistency: 2D protocols have poor slice consistency, 2D CNNs exacerbate this problem further
                                3D CNN preferred

> Unrelated (to SIE presentation) asides
Things to check: Baseline methods
                 Maybe simulate more extreme examples? (Difference in segmentation volumes v. small as %)


**NOTE** SPGR simulation equation was INCORRECT! Denominator is [1 - cos(x)*exp(...)], **NOT** [(1-cos(x))*exp(...)]  ##CORRECTION
          <REPRECUSSIONS> * Effects of __TR__ AND __FA__ not accounted for properly
                          * FA has a MUCH greater effect in changing contrast (prop. to T1-W)

>        Proper showcase: http://mriquestions.com/spoiled-gre-parameters.html *EXCELLENT for visualising effect of parameters on image contrast*
         Other simulation points: The parameter ranges seem insufficient to provide a large array of contrasts
                                  With equation correction: * FA range should be reasonable
                                                            * Should consider extending TR considerably (UB ~800)
                                                            * TE does not display signs of great variance (5 - 50: Minimal difference)

@Solution: * Delete all ongoing cluster jobs [Done]
           * Re-simulate volumes for reasonable range _WIP_: * TR: 50 - 800
                                                             * TE: 5 - 40 *Consider freezing this at a first approach*
                                                             * FA: 5 - 75
           * Train network with new images
             * Have interesting new paradigm: Because of variability in FA now have some T2s-W in images
                                              Will network be able to carry out segmentations (Baseline)?
                                                Consulted with Mauricio: Multi-modal networks should exhibit same (if not better!) performance
                                                with multiple modalities (Due to less overfitting)





# Log: 06.11.19
Aims: Cross-validation work
      SIE slides
      Check new SPGR simulations

Location of (current) best performing MPRAGE inference images: </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run>
  *LongerFC_Run Physics (FCL 20)*

Did some work on cross-val: </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run/cross_val{...}>
Running corrected spgr locally: </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR/Corrected>

FINALLY went back and started documenting how on earth PGS maps were created: See ~/bin/PGS_procedure.sh
  Corrected some scripts (detailed in PGS_procedure.sh)
  Proof of matching: Re-ran through all steps, PGS maps match those on DGX1! *Last step was getting rid of infinites, accounted for 10 bit diff in sub_9*
    *Check with fslstats <FILE> -R, should be [0.000000 1.000000]*
    Accordingly, have edited <label_copier.sh> to look at these NEW ProbMaps: Directory includes Train + Inf!
                                                                              Infinites have been purged!
    NEW PGS: </home/pedro/Project/Simulations/AD_dataset/MPMs/PGS_RelevantPaper_Inf/ProbMaps/SS_ProbMaps>  ##PGS
Some additional iSMORE work: Sent email to paper author with some queries, started populating slides with overview

> Asides
RNN/ LSTMs intro: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
NNUnet reading: https://arxiv.org/pdf/1809.10486.pdf





# Log: 11.11.19
Aims: * Practice + give MICCAI presentation
      * Cross-val directory + experiments setup

**[Cross-val]**
Need Labels for "standard" Inf. images: * This is because all experiments have relied on same Train/ Inf split
                                        * Need to: 1. Created touch templates for inf images
                                                   2. Transfer from DGX1 to local
                                                      [</data/Simulations/LowTD_MPRAGE/touch_templates>]
                                                   3. Run: <copier.sh> -> <label_copier.sh> in relevant directories to create labels accordingly
                                                      [/data/Labels/MPRAGE_LowTD_Inf_templates]
                                                   4. Transfer labels back to DGX1
                                                      [/raid/pedro/Labels/Physics_Gold_Standard/Labels/Inf_Labels]


Need to set up directories for carrying out CV analysis: * Parent directory: </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run/>
                                                         * Random sampling: [24, 14, 15,  6, 26, 18],
                                                                            [ 7, 16,  4, 11, 21, 12],
                                                                            [ 0,  3, 10, 17,  1, 19],
                                                                            [25, 23,  5,  8, 13,  9]
                                                           * Need to make sure these subIDs are excluded, per row for corresponding CV
                                                           * Since historically have separated Inf + Train images, have to pass both dirs now
                                                             *Since standard analyses will still rely on that first split*
                                                             *Worth randomising this whenever a job is run?*



Inf MPRAGE touch templates: </data/Simulations/LowTD_MPRAGE/touch_templates>
Cross-validation (initialisation query): https://stats.stackexchange.com/questions/352253/should-i-use-the-same-weight-initialization-for-each-fold-in-cross-validation
Training tmux: 3 (Unordered Physics, LongerFC, MPRAGE) + 4 (CrossVal for MPRAGE)
  **MOTIVATION**: Current best performer is
Had to re-run </home/.../FullSampling_SPGR/Corrected>: Incorrect handling of validation set + missing images
                                                        Former due to non-singular validation image
                                                        Latter due to non-SS of images with too short a name (Singular subID, singular FA)





# Log: 12.11.19
Aims: * Cross-val: Run All 4/5 [DONE]
      * Finish pytorch tutorial _WIP_
      * Implementation of cosine learning rate decay (+ with restarts?): https://arxiv.org/abs/1608.03983 [DONE]
Reminders: * Address diffusion data
           * Mark's linked pytorch tutorial: https://course.fast.ai/part2

> Asides
Singular Value Decomposition: http://andrew.gibiansky.com/blog/mathematics/cool-linear-algebra-singular-value-decomposition/
Unsupervised learning generals: https://medium.com/machine-learning-for-humans/unsupervised-learning-f45587588294





# Log: 13.11.19
Aims: TBD

[Cross-Val]
Should *obviously* run CrossVal on Baseline: </raid/pedro/Experiments/SS_DDDX_ProbMaps_Baseline_MPRAGE_3D>
**NOTE** Loss_border parameter is **NOT** used: Has to be explicitly called in a Crop_Layer in the application
          This is only the case in the regression application

[Paper review]
> Pulse Sequence Resilient Fast Brain Segmentation: https://arxiv.org/pdf/1807.11598.pdf
Aims: Design a NN that is able to segment images from scanners/ protocol parameters it is NOT privy to
      NOT the same aim as us: * They want a network that can just be applied to a new dataset and generalises well  #CALM
                              * We want a network that can be ROBUST to the changes, i.e.: Reverse engineer the true segmentations and decouple
                                 them from the imaging component

[Methodology evaluation]
- Decided to investigate success of method by running inference with range of parameters:
- </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Physics/Corrected/Inference/Extreme_tests>
  - TR800, TR050 are directories where TR remains constant for parameters passed, but FA is allowed to change
  - Nudged param. files: * Investigate whether network can interpolate at inference time
                         * [Params_tester800_nudged]: 1.1x the parameter values for TR800, FA33 (GT)
                         * [Params_tester800_nudged_OOB]: 2x the parameter values for TR800, FA33 (GT)
                         * [Params_tester800_nudged_VOOB]: 5x the parameter values for TR800, FA33 (GT)
    - Points of interest: * nudged_OOB provides better segmentation (TR1600, FA66) than (TR800, FA61)
                          * Suggests the network can interpolate, but doesn't know how to deal with OOB values
                          * If value is chosen that "matches" trained value then network likely (?) deals with it in a specific way?

[CDMRI Data]  ##DIFFUSIONDATA
- Did not address this in a previous log, but diffusion data has been downloaded: </data/Downloads/CDMRI>
  - Subfolders contain information relating to individual subjects *11-15*
  - Most information logged contained in <Log: 01.11.19>  *Address this!*

> Asides
- NiftyNet parameter description (Borders): https://github.com/NifTK/NiftyNet/issues/103
  - Border: *how much of the patch to ignore when doing inference. In other words, it controls how much overlap each inference patch has. The final result at a given voxel will be given by an inference iteration where that pixel was in the central region (not in the border)*
            i.e.: Take inference patch, calculate inference for patch with trimmed borders, move onto next adjacent location, repeat
            This does mean that the surrounding borders of size == border_value of a volume will NOT be calculated
              *Since they always lie in the borders*
- Border of around 8 seems to increase smoothness somewhat without large increases in computation time
  - Non-insignificant gains seen with border = 16, miniscule differences seen with border = 32 *Stick with 16 then?*  ##INFERENCEBORDER

- Relation between inhomogeneities and T2/ T2star: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2799958/pdf/10.1148_RG.295095034.pdf (Page 1434, top right)
- B0 inhomogeneities: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2802018/#S9title *Section 2?*
  - Maybe bias correction corrects for B0 inhomogeneities?





# Log: 14.11.19
Aims: - Continued extreme param tests
      - Monitor ongoing tests (cross-vals + Baseline + strats)
      - Meeting notes

[Methodology evaluation]
- Points of interest: * nudged provides slightly different, viable, segmentation *Suggests network can interpolate*
                      * nudged_OOB provides better segmentation (TR1600, FA66) than (TR800, FA61) *Suggests network understands parameter relations*
                        *Furthermore: Suggests network can work with OOB values*

[Other Experiments]
- Running Baseline on Corrected SPGR data </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Baseline>
  - **Reminder** Check inference on low contrast image first: Will give most interesting insight *TR800, FA33*

- Re-do of unordered, stratified, physics, FCL=20, DenseDice
  - Consider adding SELU activation functions instead of RELU (SELU ranked highly)
  - **NOTE** Already was using SELUs: Have renamed directory accordingly -> </home/pedro/Project/SegmentationNetworkExperiments/DGX1/Stratification/Physics/unordered_FC20_SELU>
  - Actually try with DDDX! Have done this and is the current GS  ##RESUME

_Consider_ - (Multi-Physics) SELU with FCL 20
           - Multi-Physics with FCL 40
             - **NOTE** Actually completed training for this! </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run_2>
              *Running Inference*
           - Stratification with larger batch_sizes: Better to stratify across multiple realisations than patches?
             - Can fit up to 14, will go for 8, patch size of (64, 64, 64)
             - </raid/pedro/Experiments/Stratification_Tests/Unordered_Physics_LongerFC_SELU_BS_N>


**[Experiments monitoring]**                               _Inference_, [Training]

- GIF segs.      [SPGR: 0%]    </home/kklaser/Pedrinho/MPRAGE_121_extra_Inf> [CLUSTER]                                                           <NO>
                               *Queue by the end of the day: 14.11.19*

- Physics segs.  _MultiPhysics FC40_    </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run_2/Inference> [DGX1: tmux 5]                     _WIP_
                 [MPRAGE MultiPhysics FC20 SELU: 10%]    </raid/pedro/Experiments/Multi_Test_Physics/Joined_DDDX_selu_LongerFC> [DGX1: tmux 2]   _WIP_
                 [MPRAGE MultiPhysics CrossVals FC20: 4/5]    </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run/cross_val{}> [DGX1]       _WIP_
                 [MPRAGE MultiPhysics Sep. FC20: 10%]    </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run_Sep> [DGX1: tmux 1]            _WIP_

- Baseline segs. [SPGR: 80%]   </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Baseline> [LOCAL]                           _WIP_
                 [MPRAGE CrossVals FC20: 5/5]   </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Baseline> [LOCAL]       **[DONE]**


# Jorge meeting:
Mention: * New SPGR volumes (Show?)
         * Mention interpolation tests (Show?)

|Diffusion data: Low res, might be difficult to simulate
                Approximations to diffusion method can be made, worth it?

Jog paper: Pulse Sequence Resilient Fast Brain Segmentation
           Interesting ideas: * Acknowledge difficulty in approximating imaging given so many confounding variables w/out Full Bloch sims.
                              * Get around this by simulating with approximate signal equations with variables that represent all unknowns
                                * Estimate these variables by applying GMM to image to obtain average signal values for CSF, GM, WM
                                  * 3 eqns, 3 unknowns: Use variables for simulations

> Notes
Work on uncertainty: * Can propose sequence choices based on constraining uncertainty
                      * Need to make sure uncertainty is encoded for *Message Richard about this*
                      * Also ensures that network isn't penalised for images with poor contrast *Not a problem inherent with the network, rather the image*
                     * Can then do contour plots of uncertainty
                     => Should work for Journal + MIDL submission
Disregard diffusion for now: * Could lead to dead end *+ work likely to be put under a lot of scrutiny*





# Log: 15.11.19
Aims: - Experiments monitoring
      - Uncertainty review + begin implementation (Richard)

**[Experiments monitoring]**                               _Inference_, [Training]

- GIF segs.      [SPGR: ?%]    </home/kklaser/Pedrinho/MPRAGE_121_extra_Inf> [CLUSTER]                                                           _WIP_
                               *Queue by the end of the day: 14.11.19*

- Physics segs.  _MultiPhysics FC40_    </home/pedro/Project/SegmentationNetworkExperiments/Physics_All_white_Bias_noise_blur> [DGX1]            _WIP_
                 [MPRAGE MultiPhysics FC20 SELU: 10%]    </raid/pedro/Experiments/Multi_Test_Physics/Joined_DDDX_selu_LongerFC> [DGX1: tmux 2]   _WIP_
                 [MPRAGE MultiPhysics CrossVals FC20: 4/5]    </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run/cross_val{}> [DGX1]       _WIP_
                 [MPRAGE MultiPhysics Sep. FC20: 10%]    </raid/pedro/Experiments/Multi_Test_Physics/LongerFC_Run_Sep> [DGX1: tmux 1]            _WIP_

- Baseline segs. [SPGR: 80%]   </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Baseline> [LOCAL]                           _WIP_
                 [MPRAGE CrossVals FC20: 5/5]   </home/pedro/Project/SegmentationNetworkExperiments/FullSampling_SPGR_Baseline> [LOCAL]       **[DONE]**


Memory saving: /home/pedro/NiftyNet-5/niftynet/engine/application_driver: os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' [Line 71]
Running post-processing pipeline on MultiPhysics FC40
MultiPhysics stratified FC20 SELU is current GS (WM: 1st, GM: 3rd, CSF: 1st)

**NOTE** ~/NiftyNet-5/niftynet/contrib/harmonisation/stratified_physics_segmentation_application.py/stratified_physics_segmentation_application.py:
          Did not debug issue related to increased batch size


# Log: 18.11.19
Aims: - Uncertainty implementation + testing
        - Include reading up on subject

> Uncertainty in DL
- Two types: [Epistemic] Uncertainty in the model, due to lack of training data
             [Aleatoric] Uncertainty in the data caused by noise
              _Heteroscedastic_ Uncertainty relating to the input data, where noise is NOT constant for every x
              _Homoscedastic_   Uncertainty relating to the task, where noise is constant PER task for all x

- In my work we are interested in _Heteroscedastic_ uncertainty since varying contrasts lead to varying uncertainty
  - Follow the work of https://arxiv.org/pdf/1705.07115.pdf *[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics]*
  - Define the joint segmentation Loss as: L = (1/var)*CrossEntropy + log(sigma)
    - Alter this slightly to have SPARSE CrossEntropy AND include DenseDice loss









# Log: 19.11.19
Aims: - Continue uncertainty work
        - Check network output, run preliminary inference

Network predicted uncertainty for all tissue classes combined

Modifying handler_network_output for now (Start)
Check the FA 5s!: Redid SS for these by renaming files: FA_5 -> FA_05
Consider doing JUST GM segmentation to check for feasibility
  Check to see if dice viable to include: Maybe just for segmentation task?

# Jorge meeting
- Separate uncertainty on per tissue class basis
- Be careful with final activation layer: Should not use relu because log(sigma^2) can (and will be) negative
  - Can also consider making sure the network learns sigma^2: Then can ue relu since variance can never be negative in this case
  - Using a selu while it might seems to solve the issue doesn't: This is due to the tapering off at very negative values (which are observed)





# Log: 21.11.19
Aims: - Continue uncertainty work
      - ...

[Uncertainty work]
> Current issues
Training on HighRes3DNet_hetero_multi: * Very unstable
                                       * NaNs arise almost instantly: 1. Corrected network since acti_func = None in last seg. layer
                                                                      2. Tried correcting CrossEntropy: Zeros -> NaNs (See asides for today)
                                                                        2a. **NOTE** Need to _SOFTMAX_ first! (Part of softmax_with_CE implementation)
                                                                      3. Currently investigating losses
                                                                        3a. Trained only with normal CrossEntropy: [Worked]
                                                                        3b. Tested use_feature_loss flag: [Worked]
                                                                        3c. Clipped gradients + used combined seg. loss (DDDX): Initial instability but
                                                                            network adjusts
                                                                        3d. Clipped gradients further + used ONLY CrossEntropy loss: Initial instability
                                                                            but network adjusts
                                                                      4. Tmux terminals of training: [TMUX:1] [TMUX:2]
                                                                      5. _WIP_

(Local) 3D UNet training: * Segmentation looks good, but uncertainty does not
                          * Training in: </home/pedro/Project/SegmentationNetworkExperiments/Uncertainty_Experiments/Multi_Physics>
                          * _WIP_

> Gradient clipping code:
<!--                                                         def clipGrads(grad):
                                                                 if grad is None:
                                                                     return grad
                                                                 return tf.clip_by_value(grad, -1e3, 1e3)

                                                grads = [(clipGrads(grad), var) for grad, var in grads] -->
                                                 *after the line grads = self.optimiser.compute_gradients*

> Asides
- CrossEntropy reformulation: Add small term to prediction to avoid NaNs: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
- Final layer activations: Non-linear for class/ seg, linear for regression
                           https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer





# Log: 22.11.19
Aims: - Continue uncertainty work
      - GIF check

[Uncertainty work]
- Overnight work no fruition yet: * Bizarre looking uncertainty maps
                                  * Started new experiment: Non-SS training images to try to replicate Richard's
                                    * </raid/pedro/Experiments/Uncertainty>

- Persisting with Local UNet: * Max out at 100k iterations
                                * Altering config files to make this the max

[Corrected SPGR GIFs]
- Almost done, one missing: </--spgr_sub_14_TR_0.2000_TE_0.0050_FA_54.0--/>
  - Seems to have run into some trouble, copied file to GIF directory and re-running job </cluster/project0/possumBrainSims/GIF/SPGR_Extras>





# Log: 14.11.19
Aims: - Uncertainty work weekend runs analysis
      - PyTorch tutorial (continued)  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
      - Fast AI PyTorch (begin)

[Uncertainty work]
- Most runs failed part of the way through: OOM/ Device space issues
- No success with uncertainty in any case, very noisy outside prediction regions (i.e. GM)
  - Consider turning off bias field?
- Try No New UNet *NiftyNet implementation*: * Good performance according to Richard
                                             * Main differences are: {LeakyRelu}
                                                                     {SAME convs}
                                                                     {Reduced upFilters}
                                             * https://github.com/NifTK/NiftyNet/blob/dev/niftynet/network/no_new_net.py
                                             * Running trial run on DGX1 [TMUX: 1], </raid/pedro/Experiments/Uncertainty/NNUnet>  *Check ~17000 iters*

- Richard uses 3D Labels instead of 4D: Worth trying?
  - Need to make these based on current PGS labels: 1. Split images across -t dim
                                                    2. Add Non-GM tissues together  *0001 0002 0003*
                                                    3. Merge 3D sums with 3D GM  *In this order*
                                                    4. Argmax 4D [Non-GM GM] volume to obtain 3D GM label
                                                    5. See: </bin/4D_to_single_3D_Label.sh>  _InputDir_ _TissueID_
                                                      5b. Running in </data/Labels/MPRAGE_121_PGS/>

[GIF]
- Final job finished running </--spgr_sub_14_TR_0.2000_TE_0.0050_FA_54.0--/>
  - Extracted and organised directory
  - Downloading 4D segmentations to </home/pedro/Project/ImageSegmentation/>
    - Should replace GIF_SPGR directory with this one, or at least name new directory GIF_SPGR_Corrected
    - Need to Argmax, of course, for use in analysis

> Asides (Almost all Bash today)
- End characters based string slicing
  https://stackoverflow.com/questions/27658675/how-to-remove-last-n-characters-from-a-string-in-bash *First answer: Pierre D comment*
- Remove element from string in bash (Simple)
  https://stackoverflow.com/questions/23462869/remove-element-from-bash-array-by-content-stored-in-variable-without-leaving-a *Second answer*
- Rounding with Python3 format
  https://stackoverflow.com/questions/1598579/rounding-decimals-with-new-python-format-function





# Log:
Issue is in here!

if self.action_param.loss_type == 'DDDXUncertaintyLoss':
    joint_loss = scaled_approx_DDDX_softmax(
        prediction=seg_net_out,
        noise=unc_net_out,
        ground_truth=label,
        weight_map=data_dict.get('weight', None))
    loss = cross_entropy_dense(
        prediction=seg_net_out,
        ground_truth=label,
        weight_map=data_dict.get('weight', None))
    data_loss = tf.cond(self.use_feature_loss,
                        true_fn=lambda: joint_loss,
                        false_fn=lambda: loss)
else:
    data_loss = loss_func(
        prediction=net_out,
        ground_truth=data_dict.get('label', None),
        weight_map=data_dict.get('weight', None))

if self.net_param.batch_size == 2 and (self.net_param.queue_length == self.net_param.batch_size):
    total_feature_loss = feature_loss_func(
        prediction=features_out[0, ...],
        ground_truth=features_out[1, ...])

    def stratification_checker(input_volume):
        return tf.reduce_sum(tf.subtract(input_volume[0, ...], input_volume[1, ...]))

    alpha = 0.1
    regulatory_ratio = alpha * (data_loss / total_feature_loss)
elif self.net_param.batch_size > 2 and (self.net_param.queue_length == self.net_param.batch_size):
    feature_loss1 = feature_loss_func(
        prediction=features_out[0, ...],
        ground_truth=features_out[1, ...])

    feature_loss2 = feature_loss_func(
        prediction=features_out[1, ...],
        ground_truth=features_out[2, ...])

    feature_loss3 = feature_loss_func(
        prediction=features_out[0, ...],
        ground_truth=features_out[2, ...])

    total_feature_loss = tf.reduce_mean([feature_loss1,
                                         feature_loss2,
                                         feature_loss3])

    def stratification_checker(input_volume):
        return tf.reduce_sum(tf.subtract(input_volume[0, ...],
                                         tf.add_n([0.5*input_volume[1, ...], 0.5*input_volume[2, ...]])))

    alpha = 0.1
    regulatory_ratio = alpha * (data_loss / total_feature_loss)

else:
    print('No feature normalization')


Problem was with "loss"! It was being calculated for some reason and there were issues with dense_XENT definition in the file: Moved softmax -> sparse softmax to work with 3D label files (Same treatment as within the scaled_approx_DDDX_softmax)
Also, 3D label >>>> 4D label for some reason!
  Need to investigate this further tomorrow





# Log: 27.11.19
Aims: - AML work (Jorge meeting clarification)
      - Uncertainty work
        - Make corrections according to errors found in previous day

# Jorge meeting
> AML course
- Focus on slides, find natural ways to break up lectures for examples
- Fine to re-use code, e.g.: * Make one multi-task network to showcase
                             * Re-use to say you can scale losses
                             * Re-use to say you can normalise gradients instead of losses

> Uncertainty work
- Don't separate physics from uncertainty branch: * Physics is what allows us to decouple uncertainty from anatomy from uncertainty from imaging changes
                                                  * Baseline gives coupled uncertainty, Physics method gives anatomical (?)
                                                  * If anything, remove from segmentation (But not actually)
- Leave networks to train for a few days, clear that uncertainty training takes much longer to converge

[Uncertainty work]
Initiated/ modified a few jobs/ experiments: </raid/pedro/Experiments/Uncertainty/NNUnet>                                          [4D]
                                             1. Low epsilon mod                     *0.05 -> 0.005: After 3477 iters*
                                             2. Looser gradient clipping (1e4) mod  *1e3 -> 1e4: After 3477 iters*
                                             3. Higher learning rate mod            *1e-4 -> 5e-4: After 3737 iters*

                                             </raid/pedro/Experiments/Uncertainty/3DLabels/NoDiceClipsNoCosine>                    [3D]
                                             1. Lowest epsilon             *Zero*
                                             2. Looser gradient clipping   *1e4*

                                             </home/pedro/Project/SegmentationNetworkExperiments/Uncertainty_Experiments/NNUNet>   [3D]
                                             1. Low epsilon              *0.005*
                                             2. No gradient clipping     *inf*

> Leaves space for completely unbounded training, i.e.: Zero epsilon & Zero gradient clipping  *In 3D, at least*

[AML]
- Possible resources: https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
- Other: * Evernote notes (Deep Learning Coursera: Part 3)
         * UvA, MILA courses: Good for Slide structure + guidelines
         * ML course notebooks + slides: Good notebook guideline


> Asides
- Upsampling variations: Visualised
  https://datascience.stackexchange.com/questions/38118/what-is-the-difference-between-upsampling-and-bi-linear-upsampling-in-a-cnn
  https://tinyurl.com/rsm6bg8 *Image*
- Sparse vs non-sparse softmax_cross_entropy_with_logits explained
  https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm
- Sort directories by size:                 <!-- du -m --max-depth 1 | sort -rn -->
  https://linuxconfig.org/list-all-directories-and-sort-by-size





#  Log: 28.11.19
Aims: AML Slides: Lecture 6: Multi-Task Learning

[Resources]
GradNorm paper: https://arxiv.org/pdf/1711.02257.pdf
Multi-Task blog: https://towardsdatascience.com/multitask-learning-teach-your-ai-more-to-make-it-better-dde116c2cd40
More in depth: https://ruder.io/multi-task/
Multi-task thesis: http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf

Facial recognition image: https://scx1.b-cdn.net/csz/news/800/2017/letsfaceitwe.jpg
Ruder multi-task image (From Kendall 2017 uncertainty paper): https://ruder.io/content/images/2017/05/weighting_using_uncertainty.png

Multi-task blog: https://medium.com/@zhang_yang/multi-task-deep-learning-experiment-using-fastai-pytorch-2b5e9d078069
  Associated data: http://aicip.eecs.utk.edu/wiki/UTKFace

> Auxiliary losses
"Given the relatively large depth of the network, the ability to propagate gradients back through all the
layers in an effective manner was a concern. One interesting insight is that the strong performance
of relatively shallower networks on this task suggests that the features produced by the layers in the
middle of the network should be very discriminative": https://arxiv.org/pdf/1409.4842.pdf (GoogLeNet paper, page 6)
https://stats.stackexchange.com/questions/304699/what-is-auxiliary-loss-as-mentioned-in-pspnet-paperz

p-norm *Note the bracket placement!*
https://stats.stackexchange.com/questions/181620/what-is-the-meaning-of-super-script-2-subscript-2-within-the-context-of-norms





# Log: 29.11.19
Aims: - Uncertainty work: Preliminary analysis and iterating methods
      - AML: Continue, look into https://ruder.io/multi-task/

[Uncertainty work]
Jobs recap: * Local 3D, epsilon = 0.005   *20k*
(No SS)     * DGX1 3D, epsilon = 0        *29k*
            * DGX1 4D, epsilon = 0.005    *16k*

Jobs crashed as a result of NaNs: * Rapid increase in loss observed before NaNs/Infs produced
                                  * Repeated resetting of job shortly prior to instability proved ineffective
                                    * Tuning of lr also had limited effect *Especially on 4D job*

Made alterations to job-related files: * Made sure instance norm was turned on where relevant by replacing None with feature_normalization in block_layer
                                       * Replaced hacky implementation of 'softmax_cross_entropy_with_logits' with TF implementation
                                         * Had initially made this replacement due to perceived instabilities relating to loss
                                         * HOWEVER, issues were likely due to hyperparameter/ network choice, so OK to revert changes
                                       * Started new job with this version: </raid/pedro/Experiments/Uncertainty/NNUnet_MultiGPU> *Not actually Multi-GPU*
                                         * DGX1, 4D [tmux: 5]
                                         * Seems to perform well at first glance
                                           * Hacky loss job had odd, very slowly decreasing loss *Probably implementation error on my part*


Made **FUNCTIONING** log trimming script: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/log_trimmer.py>  [LOCAL]
                                          </home/pedro/bin/log_trimmer.py>  [DGX1]
                                          1. Accepts two (compulsory) arguments: [--event] [--retain_step]
                                             [--event] Log file to be trimmed
                                             [--retain_step]
                                          2. Every retain_step writer writes EVERYTHING to new log file *Except graph apparently, oops*
                                          3. Every other step only graph data and histograms are maintained
                                          4. See https://stackoverflow.com/questions/42774317/remove-data-from-tensorboard-event-files-to-make-them-smaller

Additional: * Wrote bash script that runs log_trimmer with retain_step == 10  <home/pedro/bin/auto_log_trimmer.sh>  [DGX1]
            * Loops through current directory searching for files with [*training/event*']   <!-- find . -type f -wholename '*training/event*' -->
            * Deletes original log file upon completion
            * Do NOT touch validation since graph data contained there and often not worth reducing size (Keep option open for future)
              * Actually, worth touching validation: Just had to maintain initial [if event_type != summary] logic to keep graphs
                **NOTE** Ended up accidentally deleting full logs: Due to [writer.close()] being in inner loop *therefore closing after EACH event*
                         Fixed: Now runs properly



> Asides
Clearing out space on DGX1: Folders downloaded/ deleted to be documented here
pip3 install -U --no-deps tqdm





# Log: 02.12.19
Aims: - Uncertainty check (Pre-lunch)
      - AML slides: continued (Post-lunch)
      - Notebook: Begin

      [Uncertainty work]
      Jobs recap: * Local 3D, epsilon = 0.005   *Did Not proceed over weekend*
      (No SS)     * DGX1 3D, epsilon = 0        *Very unstable, unrecoverable instability at ~51k*
                  * DGX1 4D, epsilon = 0.005    *Seemed stable, unrecoverable instability at ~10k*
                    * Prioritize this job since 4D maps are used
                    * Restart training just prior to instability (9k)
                    * Run inference on 9k model for preliminary evaluation

> Asides
PyTorch ResNet code explanation: https://medium.com/@erikgaas/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096
  Bottlenecks: https://miro.medium.com/max/990/1_STAR_j_lC2gsO1Kbia8PIQGHUZg.png  *Replace _STAR_ with asterisk*
               Essentially, simply a way to be more memory efficient.

Image loading from directory in PyTorch: https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
Train/ Test/ Val splits in PyTorch: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

ImageLoader: Expects a class-based subdirectory structure in passed path
             Need to construct a custom data loader to handle this

Elastic deformation: https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
Cosine gradient similarity paper: https://arxiv.org/pdf/1812.02224.pdf
Very useful blog post (Images galore): https://vivien000.github.io/blog/journal/learning-though-auxiliary_tasks.html

To include: * Cosine loss (Auxiliary tasks)
            * Examples for each
            * Ask Carole for data: Lesions (Main) + WM/ GM segmentation as auxiliary

ResNet PyTorch implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

{:.Nf}   ##FORMATSTRING


> GradNorm: https://openreview.net/pdf?id=H1bM1fZCW
- Allows for more balanced multi-task training by normalising loss functions by looking at relative task gradient sizes
- Process (At every iteration): 1. Calculate inverse learning rate for each task: [L'(t) = L(t)/L(0)]
                                2. Calculate averaged inverse learning rate across tasks: [L' = E(L'(t))]
                                3. Calculate inverse rate for each task: [r = L'(t)/L']  *Adjusts learning: Slower tasks pushed to learn faster and V.V.*
                                4. Calculate gradient L2 norm for each task for a layer: [G(t) = ||dL/dW||_2]
                                5. Calculate averaged gradient L2 norm across tasks: [G = E(G(t))]
                                6. Calculate desired gradient for each task: [G_des(t) = G*r^&]  *Alpha further adjusts task balance*
                                7. Calculate gradient Loss for each task: [L_grad(t) = |G(t) - G_des(t)|]
                                8. Calculate overall gradient loss: [L_grad = E(L_grad(t))]

- Renormalise loss weights after every step to decouple them from learning rate (Want their sum to always be constant, only relative changes)
  -                                                             [w_i = w_i / E(w)]

Pytorch implementation: https://github.com/hosseinshn/GradNorm/blob/master/GradNormv8.ipynb


> (Un)weighted cosine loss adjustments: https://arxiv.org/pdf/1812.02224.pdf
- ...
- ...
- ...
- ...
- Gradient modification discussion: https://discuss.pytorch.org/t/how-to-modify-the-gradient-manually/7483/5

TensorFlow implementation: https://github.com/szkocot/Adapting-Auxiliary-Losses-Using-Gradient-Similarity/blob/master/notebooks/1-Concepts.ipyn
  **NOTE** Main difference with blog implementation is that care is taken to ONLY apply gradient changes to SHARED layers





# Log: 03.12.19 - 06.12.19
- Working on Uncertainty + AML, contributing in {LogAbove}

# Jorge meeting
Uncertainty work: * Instance norm could be causing issues: Patch dependence even at inference time
                                                           Options: 1. Increase patch size at inference time + padding (To allow for 200+)
                                                                    1. Turn off instance norm. completely
                                                                    2. If training too unstable, consider decaying alpha normalisation (-> 0 norm)
                  * Carrying out Non-Instance Norm experiment in MultiGPU folder with: 1. Decaying epsilon: [5 * 10 ^ (-3 - (iter/2500))]
                                                                                       2. Cosine loss decay

> Asides
- Rounding output console values (e.g.: Loss, learning rate): * </NiftyNet-5/.../engine/application_iteration.py>
                                                              * Modify [_console_vars_to_str] function: *console_str = ', '.join('{}={:.6f}'.format...*





# Log: 09.12.19 - 13.12.19
Aims: - AML notebook: FINISH main experiments
        - Preferably move to Emma's notebook

Exiting script easily: https://stackoverflow.com/questions/543309/programmatically-stop-execution-of-python-script/543375

[AML]
> GradNorm
- Maybe should still initialise at 2000 to even playing field?
  - Non-initialisation leads to large instability owing to L_1 >>> L_2, therefore weights get adjusted until w1 < 0
  - Leads to further issues since network tries to optimise for negative loss by ruining Task 1
  - Currently trying to adjust by having [L_1 == sqrt(MSE)] **INSTEAD** [L_1 == MSE]

[Uncertainty]
- Running experiments for other tissues: * GM </raid/pedro/Experiments/Uncertainty/NNUnet_MultiGPU>  [tmux: 1]  *Re-run*
                                         * WM </raid/pedro/Experiments/Uncertainty/NNUnet_4D_WM>     [tmux: 2]
                                         * CSF </raid/pedro/Experiments/Uncertainty/NNUnet_4D_CSF>   [tmux: 4]

[Dealing with medical data]

Overall: https://link.springer.com/article/10.1007/s10278-019-00227-x
2.5D: https://link.springer.com/content/pdf/10.1007%2F978-3-642-40763-5.pdf (Page 283)

histogram_normalisation: * Calls histogram_standardisation at <hs.create_mapping_from_multimod_arrayfiles>
                           * compute_percentiles: 1. Create array of percentiles [perc] (Ranging from min(cutoff) -> max(cutoff) w/ 0.1 jumps)
                                                  2. Mask image (If there is a mask) and flatten
                                                  3. Call np.percentile on image: Obtain array with percentile values of image at each [perc]
                                                  4. For each calculated percentile, mu_k, calculate mean intensity across images
                                                     [mu_]

> Asides
Whitening in the command line: [seg_maths example.nii -sub $(fslstats example.nii -m) -div $(fslstats example.nii -s) output.nii.gz]
  *Capitalise -m and -s to ignore non-zeros*


# Things left to complete
Multi-Task: * Label dirty-ing to enhance normalisation performance  <N>
            * Auxiliary loss implementation  <N>
            * Slides + Examples              <N>

Medical image data: * Histogram implementation  <NN>
                    * Bias field implementation  <N>
                    * 2D vs 2.5D vs 3D  [Done-ish]
                    * Patch based training  [Done-ish]
                    * Normalisations: Batch vs Instance vs Group- Associated problems  <NNN>

> Asides
Grid creation: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
numpy.argsort: Given array, returns sorted array of INDICES relative to first array
                *E.g.: [33, 11, 22] -> [1, 3, 2] (If you replace output indices with corresponding numbers you get a sorted array)*
               https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
print(len(dataloader.dataset))

> Uncertainty
Read Kendall paper in greater detail: Actual meaning of output + Gibbs distribution
Potentially useful repo (Yarin Gal): https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb

> Selective weight restoration
Dan meeting: 20th *Have a chat with Tom beforehand to see if he's managed/ assess feasibility*
Handler_model: Insight into model restoration
Create physics network with NNUNet as preparation: Download current version of NNUNet from DGX -> Local and make changes from there

> Uncertainty notes (continued)
- According to the Kendall paper, in a classification/ segmentation setting the probability dist. == Gibbs
  - This is because SoftMax(x) == Gibbs(x)
  - Therefore "variance" is actually just the Gibbs temperature
  - Therefore true measure of uncertainty is found by calculating the entropy https://en.wikipedia.org/wiki/Entropy_statistical_thermodynamics
    - In application:                       [B = SoftMax(seg_output / exp(unc_output)]
                                                   [Entropy = -(B x log(B))]
  - Also means that there is no need to "cap" the uncertainty, since the values are NOT variance
    - Re-training from 15031: 0.05 decaying epsilon: Saving models every 100 iters: Training is unstable and prone to segmenting skull  *lr 5e-5*

> Saving terminal output to file while MAINTAINING output: tee (See stratification folder history in tmux 3 for use cases)
- <COMMAND> | tee <OUTPUT>  *Typically a .txt file*  ##TEE ##TERMINALOUTPUT ##MULTIPLEOUTPUTS



Notebooks!!!!! /home/pedro/Project/PhD_side_projects/ML/MachineLearning2019/Materials/Week 8b - Neural Networks/Notebooks *Untitled(s)*


# Upgrade report
Image references: Precession.png https://my-ms.org/mri_physics.htm [Accessed: 14:58, 13/01/2020]
Good uncertainty comparison paper: https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf
Uncertainty diagram (Three plots): https://towardsdatascience.com/what-uncertainties-tell-you-in-bayesian-neural-networks-6fbd5f85648e

Main to-dos: * Background sections: MRI + Deep Learning (Need to decide on level of detail) *E* <DecideIfNecessary>
             * Intro re-write [Mostly done]
             * Lit review completion [Major sections mentioned] <Detail> <PhysicsAndDLMergeSection> <DeeperLookIntoUncertainty> <CheckStandardisation>
             * Paper section extension [Mostly done]
             * Conclusions + Future work *M*
             * Bibliography [On track]

# Literature review
Incorporating physics in DL (Part I): https://arxiv.org/pdf/1711.10561.pdf
Physics-informed generative models: https://arxiv.org/pdf/1812.03511.pdf
Treatment planning MR simulation: https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.4896096
Artefact correction MR simulation (POSSUM): https://www.ncbi.nlm.nih.gov/pubmed/26549300
SPGR equation complete algebrization: https://iopscience.iop.org/article/10.1088/0031-9155/55/15/003/pdf
Kwan et al.: https://www.researchgate.net/publication/2723037_An_Extensible_MRI_Simulator_for_Post-Processing_Evaluation
Zach uncertainty paper: https://arxiv.org/pdf/1907.11555.pdf
Unet explanations: http://deeplearning.net/tutorial/unet.html
Ernst equation: https://en.wikipedia.org/wiki/Ernst_equation

# Standardisation techniques
W-score: https://tinyurl.com/tsz7yqy
> Limits: Requires healthy + pathological data for calculation of w-scores
>         Assumes homogeneity within protocols

(Percentile) Histogram standardisation: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.102&rep=rep1&type=pdf (Nyul paper)
> Limits: "(...) the histogram matching method always produces a different standardization, depend-ing on the spatial coverage of the histogram. This shortcoming is acute inmulticenter data. Protocol differences may induce nonlinear spatial biason images that could lead to difficulty in defining thespatial coverage ofthe histogram." (Discussed in W-score paper)

Gaussian modelling histogram standardisation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5333904/pdf/pone.0173344.pdf (Hellier paper)

KL divergence: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1398484 (Weisenfeld)

Voxel or vertex-wise standardization methods: (Hedeker et al., 1991; Jack et al., 1997; Moerbeek et al.,2003; Teipel et al., 2012; Weiskopf et al., 2013)
> Limits: Assumptions of linear-type changes with protocol difference


Atlas-based methods: vanErp2015, Ashburner2000

Random forest methods:

Deep learning methods:

# Uncertainty

# MR Simulators
TR = TI + N*tau (slice imaging time) + TD (delay time, optimal at zero)

mv AD_dataset -> data

Conclusion + Future work: * Main goal is to achieve complete generalisation
                          * Work towards that in steps: * Multi-Sequence: T1 + T2
                                                        * Multi-parameter
                                                        * Uncertainty work to pinpoint predictions
                                                        * More complex simulation work
                                                        * Image-type specific batch norm

> Asides
[Calculating mean and saving to a text file]: fslstats -t <FILE> -m >> <OUTPUT.txt>
[Keeping only even lines of a text file]: awk 'NR%2==0' infile > outfile
[Parallelising bash scripts]: https://www.gnu.org/software/parallel/
[Connecting to UCL cluster] Modified ssh config: * storm -> jet        *Bouncer*
                                                 * comic2 -> comic100  *Login node + command*
[FLAIR sequence] * FLuid Attenuation Inversion Recovery simulation possible, follows standard IR:
                 * http://mriquestions.com/choice-of-ir-parameters.html
                 * http://mriquestions.com/t1-flair.html (T1 + equation)
                 * http://mriquestions.com/t2-flair.html (T2)

Changes: * Uncertainty figure (Get from Richard paper)   [Done]
         * Error bars on stratification figure
         * Make "those" equations more mathematical      [Done]


# Log: Back to research: Week beginning 03.02.19




> Asides:
Converting "stratified" files back to original file name:
[for i in $(seq 0 26); do mass_renamer.sh $PWD sub_*_${i}_ sub_${i}_; done]

> Parameter corrector (stratified_shuffler):
[for i in /raid/pedro/Data/MPRAGE/LowTD/Standard/renamed_BS4/SS_MPRAGE_sub_*; do
  filename=$(basename $i)
  echo ${filename}
  OG_param=sub_${filename:19:100}
  mv /raid/pedro/Data/MPRAGE_params/LowTD/Standard/${OG_param} /raid/pedro/Data/MPRAGE_params/LowTD/Standard/renamed_params_BS4/${OG_param:0:3}_${filename:14:4}_${OG_param:4:100}
done]

> Editing max images shown in tensorboard:
  /home/pedro/NiftyNet-5/niftynet/io/misc_io.py

> Other progress:
  Stratification fixed to copy file to new destination instead of resorting to renaming
    This is particularly problematic if there is an issue: Files need to be re-uploaded to DGX or copied from another directory
  Kerstin MultiRes work: Functioning implementation (Pretty quick)

Stratification problems: Precision was culprit!!! [1 - (1/3 + 1/3 + 1/3) =/= 0]
                         Solved by making stratification_checker do: (L1 + L2) - (L3 + L4) == 0 *No recurring decimals involved therefore no rounding!*





# Log: 10.02.19 - 14.02.19
Aims: - Finish DGX2 setup
      - Tune parameters to run Strat. MPRAGE

[DGX2 setup]
- Lots of tinkering with Dockerfile
  - EVENTUALLY got there: * nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
                          * tensorflow-gpu==1.15

[Stratification: MPRAGE]
- Tolerable hyperparameter choices: * (112, 112, 112) for data + labels
                                    * Batch size == 4

CUDA 10 -> 9 in dockerfile
tensorflow version: 1.10
Simon Prince book: Good for all things computer vision


# Upgrade presentation:
Get SPGR inference potentially: Would be really nice for visualisations @WontHappen
Get preliminary uncertainty results                                     @WillHaveSome
Improve contour plots (Black dots not very visible, increase fonts)     [Done]

Big -> Small
/home/pedro/Project/UpgradeReport/image_animator.py
  *Useful for animating images: brain slices in presentation's case*


# DGX2 Jobs
Strat Unc MPRAGE: ...
Strat Lenient Unc MPRAGE: <dgx2-a:/raid/pedro/Experiments/Uncertainty/Stratification_MPRAGE/Force_Stratification_Lenient_Uncertainty/Inference> [Inf]
Strat SPGR: ...

Incorporate (kernel) dropout in network as well
Stratification Uncertainty SPGR: * Does not seem to perform well at all
                                 * Try to isolate issue: 1. Train w/out stratification    [tmux1]
                                                         2. Train w/out uncertainty





# Log: 24.02.20
Aims: * Uncertainty monitoring: Especially epistemic inclusion
      * Check current results
      * Simulate further OOD volumes 1200 - 1800? [MPRAGE]

> Strat. Unc MPRAGE UncChanges [epsilon = 0]
  Fails aggressively
  Very poor segmentation performance



feature_epsilon: Internal (Specific to application) variable that modifies stratification loss term
                  *In ADDITION to stratification_epsilon term sset in the config file*
                 Set to 5e-4 initially then __Increases__ over time
                 i.e.: ENFORCES more stratification over time

> Baseline MPRAGE Uncertainty
  Currently training
  Does it make sense to train with stratification? Only seems to benefit physics method
    Seems like it will only damage Baseline

dgx2-a:/raid/pedro/Experiments/Uncertainty/Stratification_MPRAGE/Force_Stratification_Lenient_Uncertainty/Inference Inference current processed

Consider running inferences on dgx1 to allow for more training on dgx2
  Doing this: Using three GPUs
              Running 5 dropout samples

> OOD volumes
In /data/Simulations/LowTD_MPRAGE_OOD *+ params directory*


# Log: 02.03.20
Aims: * Initial dropout inference
      * Set up OOD inferences

Docker containers: laughing_shamir [Old]
                   romantic_cohen [New]  *Can access nfs drive, since it has been mounted*

Solving dropout: 1. Sample dropout masks first
                 2. Aggregate patches and apply pre-computed dropout masks on all patches
                 3. ...


# Log: 04.03.20
Aims: * Multi-dropout levels training: 0.2 *Potentially too low*
                                       0.3 *Potentially too low*
                                       0.4
                                       0.5 [Current]
                                       0.6
                                       0.7
                                       0.8

      * Preliminary epistemic + heteroscedastic analysis
      * 0.7 done ~88k

# Ongoing Log:
Aims: * Full OOD inference (Ignore .5 values for speed?)
        * 5 samples, preferably 10+
        * Ablation prob analysis
        * Check on: NoStrat job
                    NoStrat NoUnc job

> ProbAblation: </data/DGX2_experiments/Uncertainty/ProbAblation>
* Sigmoid over the sigma -> Converge -> Remove sigmoid

> Plotting
* Transforms for plotting offsets: matplotlib.transforms.offset_copy(*specify x and y offsets*)

> OOD
* Physics OOD incorrect: Parameters were missing "Standard" range
  * Baseline unaffected because only pointing to image folder: Redoing

> Dropout
* Baseline DO 2 is actually 6

> Other
* Don't use NNNet, using older implementation of UNet3D
| Purple example |

Running 5 inferences dgx2 (Physics)
Running 4 inferences dgx1 (Baseline): Space should not be an issue FOR NOW!
Remember to write: 1. Histogram code
                   2. For each bin get the probability bounds: +/- Z * sqrt(pq/n)
                   3. Obtain bound by p_ub, p_lb * n
                   4. Sum across all bins

> Runs
  * Running NoFCL_Dropout for comparative purposes
  * Getting Argmax results saved
    * Then run Argmax with whichever one of FCL/ Non-FCL worked best





# Post-MICCAI: Log: 23.03.19
Aims: * Re-organise file system to allow for multiple folds to be taken *DGX2: e.g.:</raid/pedro/Labels/PhysicsGoldStandard/Standard_All>*
      * Copy missing Inf files to DGX2, also

> SHUFFLER
Cut out prefix + extension for dataset split



Fixed issue with bad renaming of physics Params
Have issue with too much stratification?? Networks refuse to train properly





# Log: 08.04.20
Aims: * Make sure can access login node: dgx1-a
        * Can do this via: [ssh dgx1a]  *ip: 10.202.67.20x where x is the DGX*
      * Get jobs running on DGX2 using new runai system

**[RunAI Notes]** ##RUNAI
      # Quick Login: ssh dgx1a

      [Support website (Run:AI)] support.run.ai

      [Storage] * /nfs/home/pedro *Home: 500Gb limit*
                * /nfs/project/pborges *Data/ Labels/ Experiments*

      [Running scripts] * https://support.run.ai/hc/en-us/articles/360011436120-runai-submit
                        * runai submit       <JobName>
                                --image -i   <DockerImage> *NVIDIA or custom*
                                --gpu -g     <NumberOfGPUs> *INT, Typically 1*
                                --project -p <ProjectName> *Use username*
                                --volume -v  <Directory:MountName> *Directories visible to container, e.g.: /nfs:/nfs*
                                --command    <Command> *Command to be executed in container, e.g.: bash, python3*
                                --args='<ARGUMENT1>' --args='<ARGUMENT2>' etc. *Arguments to follow "command": ALWAYS AS STRINGS*
                                --project -p <ProjectName> *pedro in my case*
                                --node-type <DGXNode> *Run: kubectl get nodes to get list of nodes*

      [+ job status] * https://support.run.ai/hc/en-us/articles/360011547759-runai-logs
                     * runai get <JobName> (--output -o <FileType> --loglevel <debug|info|warn|error>) *Display job details*
                     * runai logs <JobName> (--tail -t <NumberOfLines> --since <Time> --follow) *Display job logs: More useful for monitoring*
                     * runai delete <JobName> *SE*

> DGX2: Personalised
- Currently using NVIDIA provided docker image for TF 1.x: [nvcr.io/nvidia/tensorflow:20.03-tf1-py3]
  - Can shift to using custom docker image via dockerhub in future *See Jorge email for more details*
  - Running from [/nfs/home/pedro/Stratification_MPRAGE_folds]
  - **NOTE** Finally managed to run job!
             Had to: 1. Mount nfs *Couldn't find bash file error*
                     2. Adjust absolute paths *Couldn't find data*
                     3. Alter bash file to exclude GPU line *SE*
                     4. Add arguments properly *--args='ARGUMENT' NOT --args 'ARGUMENT'*
                                               *--command only takes ONE argument, script is added as --args*

> Current jobs
[Standard MPRAGE Stratification w/ corrected physics params]
runai submit niftytestpedro
      -i nvcr.io/nvidia/tensorflow:20.03-tf1-py3
      -g 1
      -p pedro
      --command bash
      --args='/nfs/home/pedro/Stratification_MPRAGE_folds/qsub_Stratification_MPRAGE_folds.sh'
      --args='/nfs/home/pedro/Stratification_MPRAGE_folds'
      --args='train'
      -v /nfs:/nfs





# Log: 08.04.20 - 10.04.20
Aims: * Keep queuing jobs on DGX1a
      * Run TTA
      * Run Noise model
      * Tensorboard visualisation of jobs on DGX1a
      * QOL scripts
      * Wiki for running NiftyNet on DGX1a

> Noise model [hetero-noise]
- NaNs keep popping up
- Not sure why: 1. Loss function? **
                2. Stratification *Reduced already*
                3. Nans check: Get rid of it? (Gradient clipping might rectify issue anyway)

> QOL scripts
- runai submission [Done] <submitter>
- runai job list (user specific) [Done] <qstat>  *No args: List, One arg: Job names, Two args: Assumes second arg is job name and "gets"*
- runai logs checker [Done] <logs>

> Other
**NOTE** Game changer: Can rename bash scripts to drop extension!

docker pull nvcr.io/nvidia/tensorflow:20.03-tf1-py3

runai submit niftytestpedro -i nvcr.io/nvidia/tensorflow:20.03-tf1-py3 -g 1 -p pedro --command bash --args='/nfs/home/pedro/Stratification_MPRAGE_folds/qsub_Stratification_MPRAGE_folds.sh' --args='/nfs/home/pedro/Stratification_MPRAGE_folds' --args='train' -v /nfs:/nfs

:%s/\/raid\/pedro/\/nfs\/project\/pborges/g





# Log: 16.04.20
Aims: * Be able to create, tag, and push personal docker containers for use on DGX1a [Done]
        * Update QOL scripts to allow for personal containers *submitter*            [Done]
      * Tensorboard visualisation of jobs on DGX1a                                   <LolNo>
      * Wiki for running NiftyNet on DGX1a (continued)                               [Done]
      * JS applet for amigo website: Loading in nifti files = manipulation           <No>
      * Fix TTA + submission on DGX2                                                 <No>

> Personal container work
- Successful after a few hitches: 1. Created Dockerfile </nfs/home/pedro/DockerFiles>
                                    1a. Based on nvidia TF container: FROM nvcr.io/nvidia/tensorflow:20.03-tf1-py3
                                      1aa. Need [FROM nvcr.io/nvidia/tensorflow:20.03-tf1-py3] *or some other base container*
                                      1ab. Need to specify user + group args to not get permission denied errors
                                        * *I.e.: ARG USER_ID; ARG GROUP_ID*
                                        * *RUN addgroup --gid $GROUP_ID <USER>*
                                        * *RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID <USER>*
                                      1ac. Files/ directories **NOT** visible ordinarily, need to be created/ moved within DF
                                        * *E.g.: mkdir <DIRECTORY>; COPY <FILE> <DIRECTORY>; WORKDIR <DIRECTORY>*
                                      1ad. Then can run commands involving files
                                        * *E.g.: pip3 install -r requirements.txt*

                                  2. Build
                                    2a. docker build . --tag <SERVER_IP>:32581/<USER>:latest
                                        --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)

                                  3. Push
                                    3a. docker push <SERVER_IP>:32581/<USER>:latest *tag*

                                  4. Run as normal *E.g.: with submitter: Pass "personal" option instead of "nvidia"*

> Script updating
- Updated submitter to support new changes: Pass "nvidia" or "personal" as 2nd argument to specify image to be used for job

> Tensorboard visualisation
- Nothing seems to work
- Tried submitting separate job that calls tensorboard using standard docker images with zero gpus
  - Followed by standard cluster port exposing procedure
  - Cannot find port locally, however
  - https://support.run.ai/hc/en-us/articles/360011813620-Exposing-Ports-from-Researcher-Containers
    *Suggests that this feature has not been implemented yet??*





# Log: 17.04.20
Aims: - Tensorboard visualisation
      - Dr. Ivor Simpson email + inform relevant people
      - TTA!

> TTA
</raid/pedro/Experiments/Uncertainty/Stratification_MPRAGE_TTA_folds3>
</nfs/home/pedro/TTA_MPRAGE_folds3>

- Transfer code to home folder of DGX1a
- Fix code: * Currently not consistent with stratification approach
            * Randomises transform every time therefore transforms within batch are inconsistent
            * numpy seed: Unfruitful
            * Seeded array method: Could work, but need way to get ID to index array!
            * Could abstract logic by one level? *randomise method in rand_rotation gets called independently?*

> Tensorboard
1. Run **INTERACTIVE** job on dgx1a *Create tensorboard script in advance: <tboard.sh>*
2. In job run: [hostname -i]  *DGX1a*
3. Take output hostname and: ssh -N -f -L localhost:<LOCALPORT>:<HOSTNAME>:<DGXPORT> pedro@dgx1a  *locally*
4. Navigate to http://localhost:<LOCALPORT>  *locally*


> Ivor Simpson
- Need to collect papers:
- Pedro: * ...
         * ...
         * ...
- Richard: * ...
           * ...
           * ...

> Current folds management
- Keep ALL data in one folder
- Randomly assign inference/ validation/ training using [/bin/stratification_dataset_split_fold_creator.py]
  - Pass: --restricted_infs <InfsAlreadyDone>
  -       --images_folder   <FolderContainingImages>

**[How to deal with TTA]**
- Absolute mess: * Wanted to guarantee augmentation the same for same batch because of stratification
                 * BUT seems to be nigh impossible: 1. Seeding in rand_rotation layer doesn't work (Randomises all the same)
                                                      *1a. Also, images in a batch are passed *ONE AT A TIME*
                                                    2. Seeded array is a no go: No way to know iteration (unlike sampler)
                                                    3. Doing augmentation in network (Since have full batch)
                                                       **DOESN'T** work because of no eager execution
                                                       Therefore no numpy operations, rotations for placeholder tensors v. v. hard *TFA*


# Force typing of functions Python 3.6+




# Log: 22.04.20
Aims: - Meetings:
        - Jorge meeting *Semi-weekly*
        - Dan meeting *Run:AI tutorial*
        - Pritesh meeting *Intro to uncertainty methods in DL*
      - Fix problems w/ rotation pre-processing _WIP_
 @VIP - Send papers to Ivor Simpson [Done]

> Meetings
- S/E

Pritesh
- Application of standard uncertainty methods (Dropout + aleatoric method) to ascertain uncertainty in protate images *segmentation*
  - Ambit of providing radiologists with insight on regions of higher difficulty + demonstration of efficacy (artefacts)

> Rotation pre-processing
- Issue related to anisotropy of data!
- 1 x 1 x 0.5: Augmentations are carried out in pixel space (instead of coordinate space)
  - Added benefit with patches (isotropic, no bias in any one direction)
  - Added benefit for training (Fewer patches needed per volume)

- Had to install NiftyReg to access tools: Mostly
  - Mostly followed http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install
    - BUT: git clone https://git.code.sf.net/p/niftyreg/git niftyreg-git  *Instead of git://*
            cmake <niftyreg_source> *Instead of ccmake*
            Edit CMakeLists.txt instead

- Resampling ALL images to 1mm isotropic: [reg_tools -in <FILENAME> -out <FILENAME_OUT> -chgres 1 1 1]
               </raid/pedro/Data/MPRAGE/LowTD/Standard_All>   -->   </raid/pedro/Data/MPRAGE/LowTD/Standard_All_resampled>
  - Will also have to re-do epoch images (Quicker to wait for <Standard_All> resampling then running <stratified_epoch_shuffler.sh>)





# Log: 23.04.20
Aims: - Carry out resamplings

> Resampling
- Wrote changing resolution script <change_res_111> *DGX2*
      [reg_tools -in $f -out ${output_directory}/$f -chgres 1 1 1]

- Resampling procedure
  _Images_: * From </raid/pedro/Data/MPRAGE/LowTD/Standard_All>
              * To </raid/pedro/Data/MPRAGE/LowTD/Standard_All_resampled>

  _Labels_: * Don't want to resample N1000 images when they all arise from 27 singular labels
            * Created </raid/pedro/Labels/PhysicsGoldStandard/Standard_All/Individual_subjects>: Contains only the OG 27 labels
              * To </raid/pedro/Labels/PhysicsGoldStandard/Standard_All/Individual_subjects/resampled>


> Epoch images
- Current training paradigm exhausts batches **too quickly**, especially with downsampled images
  - Batches are shuffled every epoch typically, risk overfitting + not well founded otherwise with current batch setup
  - Want batches to be shuffled every epoch while maintaining stratification
  - Do this as a pre-processing step: Want at least 5 epochs of unique batches

- Wrote <stratified_epoch_shuffler.sh> to address this
  - Similar to <stratified_shuffler.sh> with additional argument: [--epochs]
  - Basically loops through standard shuffling script N times

- Settled for 10 epochs
  - **NOTE** Inordinate amount of time for resampling means settled for less (6)

- Epochs
  _Images_: * Epoch: </raid/pedro/Data/MPRAGE/LowTD/Standard_All_resampled/epoch_stratification_BS4_resampled>  *32400 images*
            * Augmented: </raid/pedro/Data/MPRAGE/LowTD/Standard_All_resampled/augmented_stratification_BS4_resampled>  *32400 images*

  _Labels_: * Epoch: </raid/pedro/Labels/PhysicsGoldStandard/Standard_All_resampled/epoch_stratification_labels_BS4_resampled>  *32400 images*
            * Augmented: </raid/pedro/Labels/PhysicsGoldStandard/Standard_All_resampled/augmented_stratification_labels_BS4_resampled>  *32400 images*





# Log: 24.04.20
Aims: - Check on resamplings
      - Check on rotated resampling: Make sure Labels and images match!


> Augmentations
- Due to all problems associated with on the fly augmentation, compromised with pre-processing
- Further modified <stratified_epoch_shuffler.sh> with additional argument [--augmentation_flag]
  - False by default: Will proceed along epoch route
  - If string passed, then will proceed along augmentation route: 1. Loops through images
                                                                  2. For every batch randomises a rotation + flip
                                                                  3. Applies to images + labels in same manner
                                                                  4. Repeats for next batch





# Log: 26-27.04.20
Aims: - Check on data augmentation process
      - Begin jobs
      - etc.

> Resampling + augmentation
- Proceeded accordingly, code seems to work properly


> Manual checking of rotations
- Only way to automate process was to look at every Nth image/ label pair
- Downloaded some examples locally to </data/DGX2_transfers/param_check>  *Disregard directory name*
  - To this end:
                                                  <for i in `seq -w 0 4 00050`; do
                                                       fsleyes Label_Label_MPRAGE_sub_${i}.nii.gz SS_MPRAGE_sub_${i}.nii.gz
                                                       sleep 5
                                                   done>

- Sequencing in bash: `seq -w <START> <INCREMENT> <END>` *-w flag ensures that leading zeros are inserted*
- BUT had to renamed files temporarily for this because otherwise would have to know TI for each image, when what matters is the ID:

                                                  <for f in ./SS*nii.gz; do
                                                      mv $f ${f:0:21}.nii.gz
                                                   done>

                                                  <for f in ./Label*nii.gz; do
                                                       mv $f ${f:0:30}.nii.gz
                                                   done>

- Had issues initially with alignment, but seemed to rectify themselves?
  - Probably due to loading multiple images into fsleyes at once with different registration?
- Also verified params to make sure they also matched





# Log: 28-29.04.20
Aims: - Job monitoring
      - Queuing in case of shortcomings

> Pure sigma
- Stratification_MPRAGE_pure_sigma_111: Disaster <Deleted>
- Stratification_MPRAGE_pure_sigma_111_no_bias: Disaster <Deleted>
- Stratification_MPRAGE_pure_sigma_111_4: Disaster <Deleted>
- Stratification_MPRAGE_pure_sigma_111_3: Disaster <Deleted>
- Stratification_MPRAGE_pure_sigma_111_2: TBD
- Stratification_MPRAGE_pure_sigma_111_5: TBD

> TTA
- Run1: 0.05 strat [tta-19439]
- Run2: 0.5 strat [tta-run2-strat-05]
- Run3: 0.005 strat [tta-run3-strat-0005]

Progressing well: Could do with training until ~50k iterations

> Gibbs
- Run1: 0.005 unc   0.5 strat    [gibbs-sigma-111-19440]
- Run2: 0.005 unc   0.5 strat    [gibbs-run2]
- Run3: 0.005 unc   0.05 strat   [gibbs-run3-strat-005]
- Run4: 0.0005 unc  0.05 strat   [gibbs-run4-strat-005-unc-00005]
- Run5: 0.0005 unc  0.5 strat    [gibbs-run5-strat-unc-00005]

Good performances overall! *Segmentation at least, uncertainties look promising as well*

Original run: * Very bizarre, only run that is negative for some reason??
              * Bias?
              * Anisotropy?
              * Will have to wait and see (inference)

> Job summary
- In summary, running multiple jobs in _Gibbs_ + _TTA_ to tune uncertainty epsilon & stratification hyperparameters
  - Ascertained that _pure_sigma_ approaches are failing, need to regularise uncertainty!
  - Uncertainty dominates over segmentation, therefore: 1. Add regularisation term to sigma *Similar to Gibbs approach*
                                                        2. Regularise loss function by downplaying uncertainty component *i.e. multiply by < abs(1)*
                                                        3. Something else?
    - Re-purpose @UncertaintyEpsilon config. param. to do this
      - If < 0.06, act as addition modulator to sigma
      - If > 0.06 act as multiplicative modulator to uncertainty component of loss
        - Contained in application: </NiftyNet-5/niftynet/contrib/harmonisation/stratified_noise_uncertainty_physics_segmentation_application.py>

> Tensorboard
- Logging of multiple individual directories didn't work
  - i.e. using [${logdir_A}:$1, ${logdir_B}] comma separated approach
    - Faced with "caching" error of some sort
- Default to passing entirety of home directory for now: **Sub-optimal!**
- **NOTE** Don't forget about <dgxboard> local script! Modified it to allow for more easy





# Log: 29-30.04.20
Aims: - Monitor various heteroscedastic uncertainty jobs
        - Pure
        - Gibbs
        - TTA
      - Update logs!
        - Thursday -> Thursday

> Job monitoring
...


python3 ~/bin/stratified_epoch_shuffler.py --images_folder /raid/pedro/Data/MPRAGE/LowTD/Standard_All_resampled/ --labels_folder /raid/pedro/Labels/PhysicsGoldStandard/Standard_All_resampled/ --physics_folder /raid/pedro/Data/MPRAGE_params/LowTD/Standard_All/ --number_subjects 27 --subject_images 120 --epochs 10 --batch_size 4 --augmentation_flag True





# Log: 06.05.19
Aims: - Jorge meeting preparation
      - Histogram creation code
      - CMIC seminar: Multi-stage Prediction Networks for Data Harmonization





# Log: 07.05.19
> OOD MPRAGE
- Had to re-simulate volumes: Only had standard [2, 5, 6, 11, 14, 25] OOD volumes
- Re-sampling MPMs for this:



> New paradigm logs
- No GPU on new paradigm
- Unscaled new paradigm: **COLLAPSE** doesn't seem to work when unscaled





# Log: 10-11.05.20
Aims: - Work on histogram outputer
      - Upload + double check OOD images (Need to make matching Labels)

> Current inference subjects (easy reference)
- 10
- 21
- 3
- 26
- 12
- 23

> Histogram application
  1. Obtains network outputs (segmentation, sigma) as LOGITS
  2. Have two separate desired outputs: * SM(X)  [1, x, y, z, 1]   *X = seg + (sig * noise)*
                                        * X      [1, x, y, z, 2]
    2a. Loop through N times adding random noise to generate more Xs
    2b. For each iteration concatenate to running variable
  3. Concatenate both large running variables to each other [1, x, y, z, 3, N]

**NOTE** Having issues with window aggregator: </nfs/home/pedro/NiftyNet-5/niftynet/engine/windows_aggregator_grid.py>







# Log: 13.05.20
Aims: - Learn regex
      - Write script for processing TTA inferences
      - Monitor noise jobs

**[REGEX]** *Python*  ##REGEX
- Means of isolating a desired pattern by means of a search sequence
- Standard syntax:                      [regex = re.compile(r'<SearchPattern>')]

From: https://regexone.com/lesson
<Letters>                 abc...
  _Only a, b, or c_       [abc]    (And only a SINGLE one of these)
  _NOT a, b, or c_        [^abc]
  _a to z_                [a-z]

<Digits>                  123...
  _Any digit_              \d
  _Any non digit_          \D
  _0 to 9_                [0-9]

<CharactersOther>
  _Any character_          .
  _Any alphanumeric_      \w
  _Any Non alphanumeric_  \W

<Repetitions>
  _m repetitions_          {m}    (Goes AFTER character to repeat)
  _m to n repetitions_    {m,n}
  _Zero or more_           *
  _One or more_            +

<Other>
  _Optional_               ?    (Goes AFTER optional character, also add on end to make non-greedy)
  _Whitespace_             \s
  _None whitespace_        \S
  _Starts and ends_       ^...$   [^START END$] (Whitespace intentional, DIFFERENT from ^ in sq. bckts)
  _Capture group_         (...)   (returns characters IN parentheses)
                                  [^(\d+)\.png$] (Looks for filenames starting with any number of digits and ending in .png, but only returns the numbers)
  _All_                   .*
  _Or matching_           (A|B)

<Advanced>
  _Lookahead_             (?=...)[whatever]  (When pattern inside this matches, start the search)
  _Lookbehind_            (?<=...)[whatever] (Only matches whatever if pattern inside parentheses matches before)





# Log: 14.05.20
Aims: - Bash script fixing
      - ...

> Optional input arguments for bash scripting  ##BASH
See: https://stackoverflow.com/questions/9332802/how-to-write-a-bash-script-that-takes-optional-input-arguments
                                         VarName=${VarNumber:-DefaultValue}]
  - E.g:                                            Var1=${1:-foo}
  - Employed in <last_iter_ini_creator>

> Passing variables in quotes for bash scripting
- 'This is an example' "${MyVar}" 'of escaping'  ##BASHQUOTES
  - Employed in <last_iter_ini_creator>

> Delete line(s) before/ after a match
- :g/<MATCH>/-1 *Delete before*
- :g/<MATCH>/+1 *Delete after*


# Log: 15.05.20
Aims: - Pure model monitoring
      - JavaScript learning
      - TTA inferences: Look at dropout as well?


**[JavaScript]**  ##JAVASCRIPT
> Beginning
- Resources: 1. CodeAcademy, FreeCodeCamp https://codeacademy
             2. FCC projects
             3. Udemy course (Get from Jonny)
             4. Mozilla dev network

- IDE, packages: VSCode (IDE)
                 Node.js (package)


> CodeAcademy
- [console] keyword: Collection of data and actions we can use in code
    <log()> method: Anything inside parentheses gets logged to console
                    *console.log(11) prints 11*
                    *Essentially equivalent to print() in Python*

- Comments: 1. // <MyComment>
            2. /* <MyMultiLineComment> */
            3. [SomeCode]/*<MidCodeComment>[SomeMoreCode]

- Data types: 1. __Numeric__: SE
              2. __String__: SE
              3. __Boolean__: SE *NOT capitalised like in Python!*
              4. __Null__: Intentional absence of a value *Assigned*
              5. __Undefined__: Absence of a value *Un-Assigned*
              6. __Symbol__: Unique identifiers
              7. __Object__: Collections of related data
  - First six are _primitive_
  - Can check types of variables using [typeof] operator
    *E.g.: typeof <MyVar>*

- Arithmetic operations: Exactly the same as in Python *Except evaluated in console log*
- String concatenation: Exactly the same as in Python *OR use string literal "Hello my name is ${variable}"*
- Attribute/ property retrieval: Same as in Python *Though property names vary, of course: E.g.: Calling .length in JS vs len() in Python*
- Methods calling: Exactly the same as in Python
  - <toUpperCase()> raises all letters to uppercase
  - <trim()> Eliminates whitespaces
- Built-in objects: Very similar to Python
  - <Math.random()>

- Variables: [var] keyword used for declaration *deprecated*
              E.g: var tester = 6
             [let] and [const] keywords preferable now
              * [let] allows variable to be reassigned *Reassignment does NOT require re-declaring*
                * Do NOT need to set a value *Will be undefined, e.g.: var tester;*
              * [const] does NOT allow for variable reassignment
                * DO need to set a value

- Increment operators: ++ & -- *Increase/ decrease variable value by one*

- String manipulation: * Standard, deprecated concatenation supported *i.e.: 'A' + 'B'*
                       * Better to use interpolation w/ _template literals_
                         * Emply __backticks__ NOT normal quotes
                         * Variable substitution identical to BASH
                           E.g.: `I own a pet ${MyVar}`

- Conditionals [if] * Statement should be wrapped in parentheses *No semicolon*
                    * Curly brackets wrapping code block
                    * [else] follows first set of curly brackets with own set *on same line*
                    * Use [else if] NOT elif

- Ternary operators: Alternative to [if], more concise *Potentially less readable?*
                     [?] to query statement truthfulness
                     [:] to separate true/ false code blocks
                      *E.g.: <MyVar> ? <DoTheTrueThing> : <DoTheFalseThing>;*


- Comparison operators: Identical as Python except for equality
                         [===] will check for equivalence AND type matching
                         [==] will ONLY check equivalence regardless of type
                         Same for [!==] and [!=]

- Logical operators: Same as in Python: [&&], [||], [!]
                      BUT reversal operator can be placed before a bool to reverse it
                        *E.g.: !<MyBoolVar>*
                        Will also convert an existing variable to false, regardless of prior type
                      Reversal operator can also wrap entire statement *Requires adding another set of parentheses, e.g.: if (!(my condition))*

- None considerations in JS: 0
                             "" '' *i.e.: Empty strings*
                             null
                             undefined
                             NaN

- Truthy and falsy assignments: * Can assign values to variables conditionally
                                  *E.g.: let <MyVar> = <SomeVar> || 'default';*
                                    The above statement will assign <SomeVar> to <MyVar> if <SomeVar> exists, otherwise will set it to 'default'
                                  AKA __short-circuit evaluation__

- [switch] keyword: * Allows for ease of checking multiple conditions on variable w/ less clutter
                      * Instead of multiple [else if] (<MyVar> === 'something'
                               ====                 [case] 'something'
                    * [switch] (<MyVar>) {....} declares the variable that conditions will be compared against *all conditions in {....}*
                    * [case] <Condition> will follows N times in {....}, where each case will be followed by a different condition
                    * [case] <Condition>: _Code block to execute if true_; break
                      * break ensures that if condition is met the loop is exited
                    * ALSO need to include a [default] _Code block to execute if no cases are true_
                      * break is still necessary here

- [function] keyword: * Behaves identically to [def] in Python
                        * E.g.: [def] <MyFunction>() {....}
                      * Passing arguments is identical
                      * return is identical

- Higher order functions: function HighOrderFunc(funcParameter) {
                                    DoSomething
                                    funcParameter()
                                    MaybeDoSomething
}
  *E.g.: HighOrderFunc times how long a given function, funcParameter takes to run*
  **NOTE** You DON'T invoke the sub_function, just pass it as a function (No parentheses)

- Function expressions: Can assign a function to a constant variable
                          - Allows for passing to other functions
                          - Less priority than declaration (Doesn't ge trun immediately)
                          *E.g.: const anonymous = function(someArgs) {....};*

- Arrow functions: Another (really pointless?) way of defining functions
                     - Declare constant variable as before with FEs
                     - Assign function parameters to it
                     - Use "fat arrows" [=>] to point at function body
                     *E.g.: const anonymous = (someArgs) => {....};*
                   Some extra quirks: 1. No need for parentheses if one argument
                                      2. No need for braces for single-line block
                                        2a. Also no need for return
                                        *E.g.: const anonymous = someArg => someArg/2*

- Scope: Variable accessibility
         Like Python: Will be able to access out of function, global, variables in function call
         Variables declared within a block code exist only within that block

- Arrays: * Equivalent to Python lists: Ordered, comma separated, contained between []
            * Indexed in exact same fashion as Python as well
            * **BUT** out of range indexing returns _undefined_ instead of an error
            * Reassignment is dependant on [let] or [const] assignment
              * [let] Can redefine elements AND whole array
              * [const] Can redefine elements but **NOT** whole array
          * Declare like you would any other variable
          * Methods similar to Python:
            * <length> method to calculate length of array
            * <push()> method to append to end of array *comma separated*  <Mutable>
            * <pop()> method to pop() as in Python                         <Mutable>
            * <slice> method to slice array *.slice(begin, end)*          [Non-mutable]
            * <shift>, <unshift> methods to remove/ append to first element
            * <indexOf()> method to obtain index of entry
            * <join>, <splice>, <concat> etc.
          * Mutable operations persist out of scope

- [for] Loops: 1. Initialisation: Declare iterator variable (All within parentheses)
               2. Stopping condition: SE *Like while condition in Python*
               3. Iteration statement: Iterator updater
               4. {CommandInLoopGoesInThese}

            *E.g.: for (let counter = 0; counter < 5; counter++)*

- [while] loops: [while] (condition) {DoSomething; counter++}
                 Ideal for when end condition is not excatly known
- [do] [while] loops: [do] {Something} [while] (Condition)

> Syntax
- End lines with semicolon [;]
- ...

> Gripes
- Null AND Undefined??
- Unnecessarily long method names
- So much damn camel case
- Semicolon line ending is a pain
- Why do template literals use backticks and not normal quotes??
- Arrow functions?? Did a 5 year-old come up with these??
- No -1 to index last element of an array??
- Nonsensical method names: shift?? unshift??
- Why is the keyword this and not self?? Makes documentation needlessly complicated, and frankly silly

> Un-gripes
- Truthy/ falsy assignments are useful





# Log: weekend + 18.05.20

- Most jobs a disaster BUT some promise! </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm> [new-paradigm-10-resumed]
  - Uses seg_focused NOT reg_seg_focused
  - Started from 39500
  - At around 96k iterations as of 23:59 on 17.05.20

> Jobs
[new-paradigm-10-resumed] + _inf-new-paradigm-10-resumed_
  - Inferences stored in </data/DGX2_transfers/pure-sigma-checks>
  - Some notes: * This job does NOT rely on regularisation in loss *i.e. log(sigma)*
                * Uncertainty starting to look reasonable
                * BUT odd behaviour at edges
                  * ln(2) around edges, also much higher than other intracranial uncertainties
                  * Look at stochastic loss to figure this out
                **NOTE** Was saving STANDARD seg_net_out, not "true" one (which includes uncertainty)
                         Have adjusted </NiftyNet-5/niftynet/contrib/harmonisation/seg_focused_stratified_physics_segmentation_application.py> accordingly
                          *Now saves f logits, sigma logits, true_seg_net_out (WM)*
                  * Segmentation NOT between 0 and 1, which is odd since sigma is quite low in background
                    * Actually due to how SM works, just because one of the class logits == 0, other class logits need to be >> 1 to have [0, 1] range
                    **Solution** Emphasise segmentation further? 15 instead of 20 *Get two jobs on this stat*


**NOTE** Should sigma be included in segmentation part of the loss??
          Created altered segmentation application where this is done:  <fiw_seg_focused_stratified_physics_segmentation_application.SegmentationApplication>

> Next steps
  1. Running job with seg_eps = 10 AND altered segmentation loss      </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_10_fiw_seg> _new-paradigm-20-resumed-from-10-110k_
  2. Running job with seg_eps = 20 AND continuing <new_paradigm> job: </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_20> _fiw-new-paradigm-10_


[TTA] jobs + _inferences_
  - Training in <Stratification_MPRAGE_TTA_111>
  - Inferences in <Stratification_MPRAGE_TTA_111/Inference> + <Inference_proper> + <Inference_proper_10>
  - Some notes: **NOTE** Big problem with inferences
                * SM segmentations are NOT between [0, 1], closer to [0.2, 0.7] in <Inference_proper>
                * BUT segmentations performed last week were between [0, 1] in <Inference_proper_10>
                  **ANSWER** DROPOUT
                  * Trained with 0.5 dropout BUT <Inference_proper>:    0.5
                                               <Inference_proper_10>  1.0
                  * Explains "over-saturation" of the IP10 inferences, which lack in non-cortical GM
                  **Solution**: Go along with Zach's paper, set first layer DO to 0.95: </Niftynet-5/niftynet/contrib/harmonisation/unet_same_multi_stratified_TTA.py>
                                1. Run inferences with altered network,   [DO 0.5] _actual-tta-inf-og_
                                2. Re-train network with altered network, [DO 0.5] _tta-3-095-do_
                                3. Continue training with altered network [DO 0.5] _tta-og-continued-095-do_




# Log: 19.05.20
Aims: - Monitor: 1. TTA jobs + infs
                 2. Pure noise jobs
      - Catch up with Dr. Simpson research (One or two papers ideally)
      - Have some prepared major questions
      - Don't forget about equation Jorge mentioned (Check pinned emails)

[Pure noise] jobs
Job listing: 1. _new-paradigm-20-resumed-from-10-110k_ </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_20>             {ResumingFromNPTen}
                * Reasonable looking segmentations, though segmentations still don't look confident
                  * Worth running a sanity check with no stochastic loss?
                  * Sub-job: _sanity_check_: Running with 1000x seg emphasis
                                             Should return VERY confident predictions, is essentially a baseline
             2. _fiw-new-paradigm-10_                  </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_10_fiw_seg>     {FromZeroFIW}
                * See above.
             3. _new-paradigm-10-resumed_              </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm>                {ResumingFromNPTen}
                * See above.
             4. _new-paradigm-5-no-reg_                </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_5_noreg>        {FromZeroFiveNoReg}
                * Quite poor segmentation, no uncertainty to be seen
                * Does still seem to be improving, probably worth training for another day
             5. _new-paradigm-5-scratch_               </nfs/home/pedro/pure_sigma_111_modulation/new_paradigm_5>              {FromZeroFiveReg}
                * Pretty awful, indicates unfettered regularisation is disastrous on performance since job [4.]
                  does not perform so poorly
                * Kill [Done]

             6. _sanity_check_                         </nfs/home/pedro/pure_sigma_111_modulation/sanity_check>                {SanityCheckZero}
                * Preliminary results after 1k resumed training iterations: Not great, segmentation still lacking
                                                                            Odd since segmentation has been emphasised so, so much
                                                                            Give it a few more thousand iterations to adjust?



[TTA] jobs + inferences
Job listing: 1. _actual-tta-inf-og_        </nfs/home/pedro/Stratification_MPRAGE_TTA_111/Inference_proper/>      {InferenceFromNegOne}
               * Same as before, look reasonable, but not as well founded of course, likely to be some over-excitation because of first layer
               * Basically, reasonable fallback in case jobs [1.] and [2.] don't pay off
             2. _tta-3-095-do_             </nfs/home/pedro/Stratification_MPRAGE_TTA_111/Run3_altered_dropout/>  {FromZero}
               * Progressing along nicely! @ 21k, starting to converge w/ [0.05 Strat epsilon]
               * Segmentations not entirely confident, but likely just need more time
             3. _tta-og-continued-095-do_  </nfs/home/pedro/Stratification_MPRAGE_TTA_111/>                       {FromNegOne}
               * Progressing along nicely! Seems to have stopped at 100k iterations [88k-100k], worth continuing post 100k
                 * Probably worth running inference @ 110k?


**NOTE** Difference between "Inference" and "Inference_proper" directories?
          No padding for proper ones, might mess with passing of physics arguments
          Might be other differences **CHECK**


> Meeting preparations
- Histograms of current best performing pure-noise model [1. 2. 3.] (windows agregator changes! Line ~120)
- TTA heteroscedastic uncertainty: Take from [1.] for now
- Gibbs heteroscedastic uncertainty


> Others
- Minor corrections made to [1a]<Inference_constructor.sh>: Was copying neg1 version of ini *Not of sh because of incorrect pointing to neg1, which is harder to*
                                                                                        *correct for because it is internal to the file*
                                                            Caused issues down the line with call to [1a]<mass_renamer_ini>
                                                              This file expected both .ini and .sh files to have same prefix
                                                                Not the case if one is neg1 and one is not
                                                            Solution: 1. Copy neg1 ini (because it is the most likely to be up to date)
                                                                      2. Rename to remove neg1 when in Inference directory

- Reverted undocumented changes to [L]<tta_variance>: Had changed file to be able to deal with single 3D volumes
                                                      **TODO** Make code more flexible to allow for these by default

- Counting of files matching a certain pattern: [find . -name "PATTERN" -printf '.' | wc -m]
                                                  [-printf '.'] prints a dot every time there is a match
                                                  [wc -m] counts how many instances of the dots there are





# Log: 20.05.20
Aims: - Metting with Dr. Ivor Simpson
      - TTA variance preliminary analysis (qualitative)
      - pure-noise FIW preliminary analysis


[Pure noise] jobs
**NOTE** Might have found the problem: * Need a SEPARATE noise sampling for each voxel
                                       **SUBNOTE** Reasoning was flawed, though softmax would change in above instance
                                                   Softmax does NOT change with constant addition, i.e.: SM([1, 2, 3]) == SM([1.1, 2.1, 3.1])
                                        * BUT should still sample separately for each voxel
                                          * More well founded? In practice other method converges to this one in limit of large N samples?

Things properly clarified in: * https://pdfs.semanticscholar.org/9253/bbb391b817457897428e2d6e39a337345f5e.pdf (Kendall's thesis, page 40)
                              * Explains correlation of logit and sigma values in relation to loss
                              <TBC>

**NOTE** Sigmas are unique for each class! Allows for softmax to vary given same noise sample across voxel logit!
          i.e.: SM([1+diff_sigma*Noise, 2+diff_sigma*Noise, 3+diff_sigma*Noise]) =/= Constant for varying noise! < **Want this**
            BUT SM([1+same_sigma*Noise, 2+same_sigma*Noise, 3+same_sigma*Noise]) == Constant for varying noise!

Network changes: * Sigma needs to be the same size as segmentation branch logits (Last dim 1 -> num_classes)   [Done]
                 * Loss needs to isolate observed class logits (use tf.reduce_max) https://www.tensorflow.org/api_docs/python/tf/math/reduce_max   [Done]
                 * Noise array needs to NOT take shape from sigma (since == seg_logits.shape)   [Done]
                 <TBC>


[New pure noise jobs]
- Going for two approaches: 1. No scaling of segmentation loss versus stochastic loss *i.e.: seg_epsilon = 1*
                            2. 10x segmentation scaling versus stochastic loss *i.e.: seg_epsilon = 10*

Job listing: 1. Post meeting <TBC>


{TTAProcessing}
- Locally: </data/DGX2_transfers/tta_infs/OG> + </data/DGX2_transfers/tta_infs/OG/rotated_outputs>





# Log: 21.05.20
Aims: - Monitor new pure noise jobs
      - Prepare for meeting with Jorge {12:30}
      - Install hard drives in Pretzel {Post-Lunch}

> Other
VAEs to model OOD sample uncertainty: - https://arxiv.org/pdf/1912.05651.pdf
ELBO explained: - https://seas.ucla.edu/~kao/nndl/lectures/vae.pdf
Nth element of N-dimensional numpy array: <ARRAY>.flat(N)

> Hard drive installation
- Mostly according to plan, some hitches regarding missing cables
- Acquired 7Tb SATA drive for desktop: Install over weekend/ Friday evening


**[Stochastic Loss]**
- Major developments here
- Loss presented by Kendall is DIRECTLY derived from {CrossEntropy}: 1. Have some predicted unaries/ logits from network [(f, say)]
                                                                     2. Class probability is [SoftMax(f)]
                                                                     3. Cast as Bernoulli distribution: [P(y|x) = p^y (1-p)^(1-y)] *For two classes*
                                                                     4. General form: [L = -Sum(ylogp)]
                                                                       4a. For two classes: [L = -y0logp0 - y1log(p0)]
                                                                       4b. Can generalise because [p1 = 1 - p0!] AND [labels are 0 or 1]
                                                                       4c. [L = -ylogp - (1-y)log(1-p)]
                                                                     5. By expanding this out: [L = - Sum(yc log(SM(f)))]
                                                                     6. Eventually arrive at equation 2.14 in Kendall's thesis!
- Deleted all (but one) <new_paradigm> folders
- Deleted all <scale_run> folders

**NOTE** Equation (12) in Kendall's paper is WRONG, it is a log likelihood NOT a loss!


- Need to MODIFY the equation to deal with SOFT labels: * All that needs to be done is convert yc to sc, where sc is the soft "probabilsitic" label
                                                        * Therefore need to sum over classes
                                                          * This is because Equation (12) explicitly define xc as the observed logit, so no need to sum
                                                            *The others would be zero, NOT the case with soft labels!*

- Create new setup: 1. New application <pure_noise_application_corrected.py>
                    2. Add losses to loss_segmentation
                    3. Get rid of seg_epsilon variable (<from user_parameters_custom> and config files)

- Set a few jobs running with different losses: 1. Original stochastic loss presented in Kendall's thesis *Equation 2.14*       {ThesisStochasticLoss}
                                                2. Negated version of stochastic loss from Kendall's paper *Equation (12)*      {CorrectedStochasticLossPaper}
                                                3. Non-negated version of stochastic loss from Kendall's paper *Equation (11)*  {StochasticLossPaper}

- Start by just creating all these losses so you don't need to redefine them in the application every time!
  **NOTE** Not actually worth it! They do not follow standard loss_segmentation layer procedure + might have some issues with fuckiness of inbuilt softmax

[Pure noise] jobs
Job listing: 1. _thesis-restart-pure-noise_     </nfs/home/pedro/pure_sigma_111_modulation/restart_thesis>      {ThesisLoss}
             2. _og-paper-restart-pure-noise_   </nfs/home/pedro/pure_sigma_111_modulation/restart_og_paper>    {OGPaperLoss}
             3. _restart-pure-noise_            </nfs/home/pedro/pure_sigma_111_modulation/restart>             {CorrectedPaperLoss}





# Log: 22.05.20
Aims: - Stochastic Loss monitoring (finally works?)
      - TTA inference (0.95 first layer DO)
      - Install 8Tb hard drive into PC

> [Pure noise] Stochastic Loss jobs
-Job listing: 1. _thesis-restart-pure-noise_     </nfs/home/pedro/pure_sigma_111_modulation/restart_thesis>      {ThesisLoss}
                 * Average performance, segmentations look decent but grey on the inside? (Probably worth doing inference)
              2. _og-paper-restart-pure-noise_   </nfs/home/pedro/pure_sigma_111_modulation/restart_og_paper>    {OGPaperLoss}
                 * Awful performance, segmentations all white, sigma non-sensical
              3. _restart-pure-noise_            </nfs/home/pedro/pure_sigma_111_modulation/restart>             {CorrectedPaperLoss}
                 * Best performance, best segmentations (~40k)
                 * Since best performing, run preliminary histogram inference: _histograms-restart_
                   **NOTE** Rememeber that window aggregator grid has been altered to allow for compatibility with histogram application!

**[SUCCESS]** Finally got reasonable results!! Turns out the corrected version of the paper "Loss" is the best!
              Continue training network and run some inferences at some point over the weekend

              * Will need to figure out how to adequately do dropout (Might work by default?)
              * Start running job with dropout enabled:
                _restart-proper-noise-095-do_    </nfs/home/pedro/pure_sigma_111_modulation/restart_do>


> Gibbs jobs
- Ignore "Original" directory logs, all others seem good
- Running job with PROPER (0.95 first layer dropout) dropout _gibbs-095-do-005-strat-00005-unc_

> TTA jobs
- Running a few inference folds with PROPER dropout _inf-tta-final-fold1_ _inf-tta-final-fold2_
- ...

> Other
- Brief chat with Pritesh about uncertainty methods + potential use cases + potential future work
- seed_parameter in networks: For inference time dropout, apparently





# Log: 25.05.20
Aims: - Monitor Gibbs job
      - Monitor Pure noise dropout job
      - Prepare OOD images for inferences
      - Look into histogram method for Gibbs Mark sent you!

[Pure noise] jobs
-Job listing: 1. _restart-pure-noise-095-do personal_     </nfs/home/pedro/pure_sigma_111_modulation/restart_do/>      {CorrectedPaperLoss}
                 * Was NOT queuing on GPU, restarted job again
                 * Problem still persisting! Keep trying for now
                 * [submitter restart-pure-noise-095-do personal ~/pure_sigma_111_modulation/restart_do/ train -1]
                 * Seems to be the case with all jobs...
[Gibbs] jobs
- Queued job with wrong parameters: * 0.5 stratification instead of 0.05 *main error*
                                    * 0.005 uncertainty instead of 0.0005 *minor error*
                                      **HOWEVER** quells doubts relating to whether or not 0.5 stratification is the optimal choice
                                                 (Since the poor performance of this job would be related to the stratification, NOT the uncertainty)

- Re-queued job with adjusted parameters: _gibbs-095-do-005-strat-00005-unc_ </nfs/home/pedro/Gibbs/005_strat_00005/>


> TTA theory
- Heteroscedastic uncertainty probably a function of augmentations, see: https://openreview.net/pdf?id=rJZz-knjz (Original TTA paper)
                                                                         https://arxiv.org/pdf/1807.07356.pdf (Guotai paper)
- Try training with only flip and noise augmentations: *18 - 26 SNR*
- Necessary to save noise augmentation?: 1. Train network with flipped + noisy images
                                         2. Test network with inference subjects that are ALSO flipped and noisy
                                         3. Final label will need to be reverted because of flip, but NOT noise
                                         4. Therefore no need to save noise transform: Non-spatial transforms do NOT alter label

> Other
- windows_aggregator_grid has been modified to allow for <histogram_creator.py> application behaviour
- fslmaths has a -log option (Useful for Gibbs jobs outputs that have exp(unc) as part of output)
- Created a copy of <stratified_epoch_shuffler.sh>, <noise_stratified_epoch_shuffler>, former only flips, latter adds noise AND flips
  - Had an issue with noise addition: * For some reason casts [x, y, z] array into [x, y, z, x, y]
                                      * Required **4Tb** of memory
                                      * Edited [add_complex_noise] function to rectify this

Vad suit + hurricane broom peng0
Shape issues with noise_stratified_epoch_shuffler.sh! *SES*
Also, edited SES to include noise: Maybe reverse this





# Log: 27.05.20
Aims: - Monitor noise/ flip image creation
      - OOD image creation
        - Maybe focus on a single subject for time efficiency purposes?

> Noise/ flip image creation
- Flip only images finished creation:

> RunAI (See ##RUNAI) ##NODES
- A few new findings: * https://support.run.ai/hc/en-us/articles/360011591500-Limit-a-Workload-to-a-Specific-Node-Group
                      * --node-type <Node>    *Flag allowing for queuing of jobs to specific nodes: dgx2-a, dgx1-1 etc.*
                        * kubectl get nodes   *List aforementioned nodes*
                        * kubectl label node <NodeName> run.ai/type=<NodeAlias> *Can assign aliases to nodes*
                                                                                *Same alias can be assigned to multiple nodes*

**NOTE** Found issue with jobs not queuing on GPUs: 1. Apparently they are being allocated GPUs
                                                    2. However, utilisation must be low BECAUSE of memory requirements of job (PS + BS)
                                                    3. Jobs have issues when trying to queue on DGX1-N but **NOT** on DGX2-a
                                                    4. Solution: Force queue on DGX2 **(for training)** using --node-type "dgx2-a" flag
  FINALLY managed to queue [Pure noise] dropout job: _restart-pure-noise-095-do_ </nfs/home/pedro/pure_sigma_111_modulation/restart_do/> FINALLY started

> OOD image creation
- [Gibbs]           No special needs.     {StratifiedEpoch}
- [TTA]             Augmented epoch       {RotationAugmentedStratifiedEpoch}
- [PureNoise]       No special needs.     {StratifiedEpoch}
- [BaselineNoUnc]   No special needs.     {NonStratifiedEpoch} *Same as stratified, with shuffling*
- [PhysicsNoUnc]    No special needs.     {StratifiedEpoch}

> Baseline networks
- Create self-contained directory for all: 1. [Gibbs]          0.0005 Unc        </nfs/home/pedro/Baselines/Gibbs>           {SpecificUnc}     [AllDone]
                                           2. [TTA]               -              </nfs/home/pedro/Baselines/TTA>             {Generic}         [AllDone]
                                           3. [PureNoise]         -              </nfs/home/pedro/Baselines/PureNoise>       {SpecificUnc}     [AllDone]
                                           4. [BaselineNoUnc]     -              </nfs/home/pedro/Baselines/BaselineNoUnc>   {Generic}         [AllDone]
                                           5. [PhysicsNoUnc]      -              </nfs/home/pedro/Baselines/PhysicsNoUnc>    {Generic}         [AllDone]

- Standardising applications: * |1e2| gradient clipping
                              * Check that physics = False
                              * lr = 1e-4
                              * 0.95 DO in first layer, 0.5 elsewhere

> Applications: https://www.diffchecker.com/diff
  * <stratified_uncertainty_baseline_segmentation_application.py>       [Done] {Gibbs}
  * <stratified_baseline_segmentation_application.py>                   [Done] {TTA} + {BaselineNoUnc}
  * <baseline_pure_noise_application_corrected.py>                      [Done] {PureNoise}
  * <stratified_physics_segmentation_application.py>                    [Pre-completed] {PhysicsNoUnc}

> Networks: https://www.diffchecker.com/diff
  * <unet_baseline_stratified_uncertainty_gibbs.py>   [Done] {Gibbs}
  * <unet_baseline_stratified_uncertainty.py>         [Done] {PureNoise}
  * <unet_baseline_stratified_TTA.py>                 [Done] {TTA} + {BaselineNoUnc}
  * <unet_same_multi_stratified_TTA.py>               [pre-completed] {PhysicsNoUnc}

- Started queuing **ALL** baseline jobs!
  - Waiting for errors to pop up...
  - Estimating: 3/5 jobs fail (Conservative)
  - Queuing command: [for direc in */; do realdirec=${direc:0:-1}; submitter baseline-${realdirec,,} personal /nfs/home/pedro/Baselines/${direc} train -1; done]
    *Execute in /nfs/home/pedro/Baselines directory*

- Issues: * Had to rememeber to set queue size to 20 in all config. files
          * A few network corrections: Mismatch between expected network output in application and network output
          * Changes to uniform sampler call in application: Expected seed image size *Not needed for baseline: No stratification*

> Other
- Bash lowercase: ${VARIABLE,,}





# Log: 28.05.20
Aims: - Monitor baselines
        - Mostly ensure that they haven't crashed from last night
      - JS resuming on codecademy
        - Maybe get round to looking into p5 or 3JS

> Baseline jobs
- [Gibbs], [TTA], and [BaselineNoUnc] crashed
  - Various application/ network naming errors, now rectified
    **NOW** Jobs now running properly
      See: _tboard-baselines_

> JavaScript:  ##JAVASCRIPT2
- Iterators: * CheatSheet: https://www.codecademy.com/learn/introduction-to-javascript/modules/learn-javascript-objects/cheatsheet
             * Allows for application of same method to multiple items in on call
             * <forEach(DoThing)> method: 1. Attaches to list and applies (DoThing) on each item
                                          2. Can define function inside *oldArray.forEach(function(iterator) {doSomething})*
             * <map()> method: Maps one array to another: *const newArray = oldArray.map(iterator => {return iterator * 2})*
             * <filter()> method: Same as .map() but filters according to some condition
             * <findIndex()> method: Return index of first element that matches condition
             * <reduce((accumulator, currentValue) => {DoThing}, NewFirstValue)> method: Returns __ONE FINAL__ value
             **NOTE** methods that have no returns will return undefined (e.g.: forEach with a console log)
                      No idea why this is relevant at all
             * <some> <every> methods: Return BOOL on array *Checks if some/ all elements in array match condition*

- Objects: * Assigned with "=", contained within {}
           * Can assign key: value pairs
           * Essentially the same as dictionaries in Python
           * Allow for mutability of keys *spaceship.numRooms = 6*
             * Complete reassignment within functions not possible though: Just assigned a different memory entry
             * Can loop through entries: for (let iterator in {Object}) {DoSomething}
           * Can create methods to objects by assigning value of a key to a function: 1. MyKey: function() {DoThing}
                                                                                      2. MyKey() {DoThing}
           * Can nest objects within objects, and even within arrays within objects

- Advanced objects: * Properties/ methods not available within individual scopes
                    * Need to call other methods/ properties using [this] keyword: e.g.: this.OtherMethod *Sort of like self in Python*
                    **NOTE** about [this]  When called within an arrow function it takes the this from the __global scope__ where the function/ object was defined
                                           When called with a normal function call it takes the this from the __context__ in which it was called *Within object*
                                           See: https://old.reddit.com/r/learnjavascript/comments/6p6hbj/how_does_the_this_keyword_work_with_arrow/dko02ja/

- Privacy: * Use underscore as prefix *Similar to Python, except no suffix*
           * setter/ getter methods: Prevents (malicious) changes
                                     If just returning a variable can/ should just use a method
           * Function factories: Functions that generate objects according to input variables

- Destructuring: * [Property shorthand] If key/ value pair share a name, can just omit the value *e.g.: name: name, ==> name,*
                 * [Assignment] If want to create a new variable that is the property of an object can omit call if property name == variable name using {}
                                *e.g.: const name = robot.name ==> const {name} = robot*

- Object built-ins: * Object.keys(<Object>): List keys of <Object>
                    * Object.entries(<Object>): List key-value pairs of <Object>
                    * Object.assign(<Target>, <Source>): Adds/ modifies key-value pairs in <Target> from <Source>

> Classes
- Very similar to Python, same concepts
- [class] keyword to create a class
- [constructor] * Treat as __init__ in python
                * Every time instance of class is created, it is called
                * Use [this] to initialise variables
- Instances: Call using new *const NewInstance = new MyClass(somArgs)*
- Inheritance: * Similar to Python
               * Child classes can inherit from a parent class
               * [extends] Child classes extends Parent class *class Child extends Parent {}*
                           All Parent methods accessible by Child class
               * [super] Calls the constructor of the Parent class
               * [static] Used to create methods that are only accessible by calling <Class> directly, NOT an instance of a class
                          Mostly utility functions, do NOT have access to any data

> Modules
- Cheatsheet: https://www.codecademy.com/learn/introduction-to-javascript/modules/intermediate-javascript-modules/cheatsheet
- Reusable pieces of code that can be exported/ imported
  - Synchronus: Waits for prior code to complete before executing
  - Asynchrous: Does not wait, defers execution (No waiting)

- [Exporting] procedure: 1. Create an object to represent the module.
                         2. Add properties or methods to the module object.
                         3. Export the module with module.exports.  *e.g.: module.exports = Airplane* {NodeJS}
                         Alternative: Use [export] keyword *e.g.: export default Airplane*  {ESSix}
                         Can export variables as they are defined *e.g.: export function MyFunc() {}*
                         Can pick naming scheme for exports *e.g.: export someExport as somethingElse*

- [Importing] procedure: 1. Import the module with require() and assign it to a local variable.  *e.g.: const Airplane = require('./1-airplane.js')*  {NodeJS}
                         2. Use the module and its properties within a program.
                         Alternative: Use [import] keyword *e.g.: import Menu from './menu'*  {ESSix}
                         Can import custom names directly

- All functions are function objects
- Multi-exports: Wrap exports in {} *export { someFunction, someString, etc.}*





# Log: 01.06.20
Aims: - Monitor baselines: Gibbs_0005 in particular
      - JavaScript: Finish Modules section on Codecademy
      - OOD Images!

> Jobs
- All seem to be progressing along nicely EXCEPT for Gibbs
- Create script to automate preliminary inference?

> Gibbs
- Awful performance
- Went back to check older inferences {DGXOne}
  - 0.8 dropout set
  - Maybe worth starting at 0.8 then decreasing?
    - Did this: </nfs/home/pedro/Baselines/Gibbs-08-do> _baseline-gibbs-08-do_
    - Running training alongside reset _Gibbs_ + _Gibbs-0005_

- _baseline-gibbs-08-do_ progressing along nicely
- Should also run completely no uncertainty Physics + Baseline jobs


> OOD
Simulated volumes: </data/Simulations/LowTD_MPRAGE_OOD_All_subjects> {AllSubjects}
                   </data/Simulations/SS_LowTD_MPRAGE_OOD_All_subjects> {AllSubjectsSS}
                   </data/Simulations/LowTD_MPRAGE_corrected_params_All_subjects> {AllSubjectsParams}
                   </home/pedro/Project/Simulations/AD_dataset/MPMs/GIF_Dil_masks/further_dils/resampled> {OGMasksResampled}
                   </home/pedro/Project/Simulations/AD_dataset/MPMs/PGS_RelevantPaper_Inf/ProbMaps/SS_ProbMaps/resampled/> {LabelsResampled}

**NOTE** Minor issue with ALL resampled Labels: * No longer between 0-1
                                                * Most values between -0.01 and 1.01
                                                * Effect is PROBABLY minor, and all experiments run under these conditions, so keep going
                                                * PURELY arises from resampling procedure
                                                * Ran minor experiment by comparing images from </home/pedro/Project/Simulations/AD_dataset/MPMs/PGS_RelevantPaper_Inf/ProbMaps/SS_ProbMaps> *Non-resampled*
                                                  to images from </nfs/project/pborges/epoch_stratification_labels_BS4_resampled/> *DGX1a, resampled*
                                                  Even re-ran resampling procedure to double check, and doubts were confirmed


> Other
- Created new script <mass_Inference_constructor.sh>
  - Runs <Inference_constructor.sh> on all sub-directories of directory passed to it
- Also modified <Inference_constructor.sh> to **FINALLY** adjust physics_spatial_window_size and patch params
  - Also added second CLA that accepts inference_iter *100k for current experiments*


> Quick to-dos
- Versions of U-Net with only ONE DO per block: **NOTE** This already happens! Only one DO per block
                                                DO applied to every second convolution
                                                In final block applied to second 3x3 layer, but NOT 1x1
                                                BUT can still consider having extended version of networks to account for reduced capacity
  - Re-train networks from 100k for ~10k iterations (Save models for OG 100k JIC) **FIX**
- Stratified epochs for TTA OOD
  - Need labels for these!
- Run Inferences on all these jobs for OOD (Done for PureNoise, Physics, Baseline, pre-FIX)
- Continue trying to make Gibbs baseline to work





# Log: 02.06.20
Aims: - Inference monitoring  [Done]
      - Epoch stratified TTA OOD creation  [Done]
        - Need to make sure labels are pushed to </nfs>  [Done]
      - TTA noise jobs: Run today!  <DoNow>
      - Gibbs baseline struggle

> OOD Inferences
- inf-baseline-pure-noise-ood: [Completed]
- inf-baseline-no-unc-ood: [Completed]
- inf-physics-no-unc-ood: *Running*

> Jobs
- Gibbs: Still trying everything to get baselines to work

- Killed jobs: _baseline-purenoise_                   {ConvergedHundredK}
               _baseline-physicsnounc_                {ConvergedHundredK}
               _baseline-baselinenounc_               {ConvergedHundredK}
               _baseline-tta_                         {ConvergedHundredK}
               _gibbs-095-do-005-strat-00005-unc_     {ConvergedHundredK}
               _baseline-gibbs_                       {BadPerformance}
               _baseline-gibbs-0005_                  {BadPerformance}

> TTA OOD
- Created </nfs/home/pedro/stratified_TTA_OOD_script.sh>
  - Runs <stratified_epoch_shuffler.py> with 30 images per subject, BS 2
    - BS 2 Because 30 not divisible by 2, and they're only being used for inference, anyway

> Gibbs jobs
- Discussion with Richard about dimensionality of Gibbs sigma
  - Seems like it is Class-dimensional
  - This means that the loss needs to be adjusted, accordingly
  - tf.nn.softmax_cross_entropy_with_logits performs summation across classes, which is NOT desirable
  - Can get around this by doing this manually: [L2 = CrossEntropy(SoftMax(logits, labels))]
                                                [L2 = -labels x log(SoftMax(logits))]
    - **NOTE** Need to add an epsilon term for stability: 1e-6 should suffice

- Training **NEW** adjusted Gibbs jobs with new loss and **N-class** sigma: _baseline-gibbs-08-do-adjusted-smaller-epsilon_  {ZeroEpsilon}
                                                                            _baseline-gibbs-08-do-adjusted_                  {OhThreeFiveEpsilon}


> Other
- Should add flag to submitter that allows for training to start from zero, even if models already exist





# Log: 03.06.20
Aims: - Jorge meeting notes
      - CDF/ histogram normalisation code
      - Monitor Gibbs jobs
      - Run noise jobs
        - Has the data been created?
        - Remember having some issues with slow creation speed

> TTA OOD
- Some files not getting created properly
- Also realised that even though num_images per subject == 30, and 0.00000 was excluded, some were still created...
  - Manually remove them
  - Re-run script
- **NOTE** Found issue with bad file creation! Presence of nans
           Correct issue within <stratified_epoch_shuffler.py> *Added remove_nans() function*
           Success! Creation progressing smoothly now: *ETA 66 hours*


> TTA Noise jobs
- Remember some speed issues with creation
- Look for relevant script and look again
- **CORRECTION** Ran jobs under </nfs/home/pedro/Stratification_MPRAGE_TTA_111/NoiseAugs>


> Other completed jobs awaiting processing:
- Job listing: * PhysicsNoUnc Baseline    [Downloaded]    </storage/UNSURE_baselines/PhysicsNoUnc>
               * BaselineNoUnc            [Downloaded]    </storage/UNSURE_baselines/BaselineNoUnc>
               * PureNoise Baseline       _WIP: 67/192_   </storage/UNSURE_baselines/PureNoise>
               * _actual-tta-inf-og_      <NotTouched>    </nfs/home/pedro/Stratification_MPRAGE_TTA_111/Inference_proper/>  {DGXOnea}
                 *How does this job differ from inf-tta-final-fold1?: Different fold only?*


> Volumetric calibrated method:
- Sample N times from model *Already doing this with both baseline and physics*
- For each voxel, construct cumulative distribution (CDF) from quantile measurements
  - Quantile == percentile: E.g.: 19 quantile -> 5% percentile split *Probably desirable quantity*
- Construct overall volumetric CDF {VCDF}
- Construct uniform distribution CDF {UCDF}
- Using [Validation] set, calculate 1D affine transform from {VCDF} to {UCDF}
  - Essentially the same as a 1D histogram normalisation: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.102&rep=rep1&type=pdf
  - Took code from </home/pedro/Project/PhD_side_projects/ML/Dealing_with_Medical_Data_Notebook_answers.ipynb>
    - Wrote dedicated script: </home/pedro/Project/Simulations/Scripts/histogram_normalisation_personal.py>  *Bare-bones, will need to modify + test later*

- **Need** to write consolidated script that: 1. Reads inference images (N-samples per)
                                              2. Computes CDFs from percentiles *How?*
                                                2a. Percentiles across N-samples
                                                2b. Summation across percentiles to get percentile volumes
                                                2c. CumSum over volumes
                                                2d. Derive uniform distribution to match against {HowToDoThis?checkScriptForCurrentImplementation}
                                              3. Calculates 1D HN transform
                                              4. Plots comparison error bar plots: 4a. Mean volume against GT volume
                                                                                   4b. Uncalibrated error bars
                                                                                   4c. Calibrated error bars

- Started writing script for this: </home/pedro/Project/Simulations/Scripts/calibrated_volumetric_estimates.py>  ##CALIBRATION
  - Investigate with example to get insight into CDFs of some subjects


> Other
- grep -v <WORD> *Excludes results that contain match*
- rsync -exclude="<FILE>" *Does not transfer any files containing match*





# Log: 04.06.20
Aims: - Gibbs...
      - Calibrated volume script
        - Test it

**[Preliminary inference investigation]**
**NOTE** Disaster: * Most Baselines look terrible
                   * PureNoise, Physics, Baseline
                   * Probably need to look at Physics inferences properly as Well

- Extreme changes to consider: * 0.8 DO across the board [Yes]
                               * Absolutely NO DO in final layer (Or at least set 0.95 there) <No>
                               * Extend network to account for DO <RunTest>

- In the meantime: 1. Compare OOD images to training images
                   2. Consider running non-DO jobs
                   3. ???

> Debugging progress
1. Realised that erroneous inference images came from specific subjects: ["Good"]  26, 23  *Good is relative, still poor compared to Non-DO PureNoise images*
                                                                         <Awful> 21, 12, 10, 3  *Awful meaning no discernible output whatsoever*
2. Those inference images have NaNs!  #NOISEIMAGES
3. Also, those images that do look passable have far too low "certainty"

Procedure: 1. Fix images locally </data/Simulations/SS_LowTD_MPRAGE_OOD_All_subjects>  [Done]
              Transferred images to </nfs/project/pborges/Nanless/>
           2. Fix the mprage generation script!                                        [Done]  *kill_nans function written*
           3. Fix alignment: -71z -> -70.75z                                            <No>   *Can fix these things in post*


**[0.8 DropOut resumes]**
- Did NOT start </nfs/home/pedro/Stratification_MPRAGE_TTA_111/Noise_Augs>
               </nfs/home/pedro/Stratification_MPRAGE_TTA_111/>
 - Did start PureNoise (Physics) + Gibbs (Physics, 0.005 & 0.0005 unc_eps)
 - Also started all baselines *Not Gibbs, need to think about that one a bit more*

> Other
- Checkpoint checker: <checkpoint_checker.sh> calls on <checkpoint_checker.py> --models_folder argument
                      See: https://stackoverflow.com/questions/38218174/how-do-i-find-the-variable-names-and-values-that-are-saved-in-a-checkpoint
                      See also: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py  *Line 56*

- Gibbs: * Changed </NiftyNet-5/niftynet/contrib/harmonisation/stratified_uncertainty_physics_segmentation_application.py> *unc passing: [..., 0]*
         * Changed </NiftyNet-5/niftynet/contrib/harmonisation/unet_same_multi_stratified_uncertainty_gibbs.py> *One unc output: i.e.: Single sigma*


- **TODO** Document above further
            Jobs, script(s), Gibbs, image comparisons
           Don't forget about OOD! Transfer relevant files to Restricted_30 from </nfs/project/pborges/Nanless>





# Log: 05.06.20
Aims: - Complete previous night's logs
      - Preliminary inference of 0.8 DO jobs
        - Probably start with PureNoise Physics
      - Check that subject {21, 12, 10, 3} images are correctly inferred
        - Check with some baseline
      - Gibbs...
      - Complete + Test out calibrated volume script
      - Udemy courses: Identify best ones


> NaN removal and transfer
- Transfer from local to </nfs/project/pborges/Nanless> complete
  - Moved from </nfs/project/pborges/Nanless> to default directory: </nfs/project/pborges/SS_LowTD_MPRAGE_OOD_All_subjects> + <Restricted_30>


**[0.8 DropOut resumes]**
- Jobs still all running, all seem smooth
- Job listing: 1. _physics-pure-noise-do_
                 Worth running prelim. inference:
               2. _test-gibbs-0005-single-sigma_
               3. _test-gibbs-00005-single-sigma_
               4. _baseline-tta-do_
               5. _baseline-baseline-no-unc-do_
                 Worth running prelim. inference:
               6. _baseline-physics-no-unc-do_
                 Worth running prelim. inference:






# Logs: 08.06.20
Aims: - Look at altered baseline jobs: 1. _baseline-no-unc-extended-default-do_
                                          Still capping visible
                                       2. _baseline-no-unc-extended_
                                          Still capping visible
                                       3. _baseline-no-unc-default-do_
                                          Speckled: Must have made mistake in network

- Did not document this last week: * Created three slightly altered versions of unet:
                                     * Extended UNet with an additional two blocks at end   <unet_baseline_stratified_TTA_extended.py>
                                     * UNet using traditional dropout layers                <unet_baseline_stratified_TTA_Default_Dropout.py>
                                     * Combined Extended UNet and traditional dropout       <unet_baseline_stratified_TTA_extended_Default_Dropout.py>

                                   * Implementation quirks: Had to alter layer_op() of UNetBlock
                                     *  Pass additional keep_prob parameter to conv_op

**TODO** Noticed that with traditional dropout was dropping out too many layers: Twice as much
         Alter network to only do this once per block: <unet_baseline_stratified_TTA_extended_Default_Dropout.py> <unet_baseline_stratified_TTA_Default_Dropout.py>

**NOTE** Restarted _baseline-no-unc-extended-default-do_ seems to reset infinitely...

> Calibrated volumetric estimates
See: https://jblindsay.github.io/ghrg/Whitebox/Help/HistogramEqualization.html
     Want to match the CDF of each image to more closely resemble that of a uniform CDF
     See notes in booklet for developments on this.





# Logs: 09.06.20
Aims: - Monitor altered dropout jobs
      - Calibrated volume script + theorising
      - ...

> Monitoring dropout jobs
Job listing: 1. _baseline-no-unc-default-do_              {Dropout}
                Problem persists [0.15, 0.85]
             2. _baseline-no-unc-extended-default-do_     {ExtendedDropout}
                Problem persists [0.15, 0.85]
             3. _xe-baseline-no-unc-extended-default-do_  {ExtendedDropoutXE}
                Collapse, dense cross entropy very unstable: Therefore no reasonable inference
                Repeat attempts have also failed

> Volumetric calibrations
See notes in booklet, will develop further

> New Jobs
Job listing: 1. _baseline-no-dropout_                     </nfs/home/pedro/Baselines/Baseline_NO_Dropout>                        {NoDropout}
             2. _baseline-no-unc-default-do_              </nfs/home/pedro/Baselines/BaselineNoUnc_DefaultDO>                    {DefaultDropout} [1.0 keep_prob]
             3. _xe-baseline-no-unc-extended-default-do_  </nfs/home/pedro/Baselines/BaselineNoUnc_Extended_Default_Dropout_XE>  {DenseDice}





# Logs: 10.06.20
Aims: - Continue to address [0.15, 0.85] segmentation range issue
...





# Log: 11.06.20
Aims: - Continued no new net investigation
      - Investigate dropout variants


**NOTE** __SUCCESS__ NoNewNet experiment produces good segmentations!!!
                     Could be NoNewNet OR adjusted loss
                     Need to check against another experiment
                     _baseline-no-dropout_ seems to have collapsed, unfortunately

> Label capper
- Completed, "proper" labels found in </nfs/project/pborges/epoch_stratification_labels_BS4_resampled/Adjusted>
  - File sizes are 2x as small, why?
  - Hopefully causes no issues

> NoNewNet
- Job listing: <Baseline_NO_Dropout> Standard UNet w/out dropout                                               [Awful]
               <Baseline_NO_Dropout_no_new_net> NoNewNet w/out dropout                                         [Excellent]
               <Baseline_no_new_net_dropout> NoNewNet w/ dropout *Models from Baseline_NO_Dropout_no_new_net*  <Pending>
               <nonewnettest> NoNewNet with STRATIFIED application                                             <Pending>
                Two possibilities: 1. Good performance: Indicates that NoNewNet is superior network
                                   2. Bad performance: Indicates some issue with application
                [GoodPerformance] Start using NoNewNet from here on out!

> Other  ##VIM
- [grep Restoring] to identify what iteration of model was loaded
- [:g/<PATTERN>/d] to delete lines containing <PATTERN>
- [ulimit -S -s unlimited] addresses argument list too longe issue





# Log: 12.06.20
Aims: - Proper experiment set up
        - Physics + Baseline

> Summary of findings: Last few days
- Found that there were issues with certain jobs' final segmentation: [0.15, 0.85] range

- Decided to investigate NoNewNet: * One level deeper than standard UNet
                                   * Same upsampling method *Resize layer*
                                   * Slightly different number of channels

- Job differences as a result: 1. Reduced patch size: 96 -> 80
                               2. Traditional dropout
                               3. KeepProb 1.0 in final layer *Speckle pattern when not the case*

> Physics: </home/nfs/pedro/Physics_NNN/>       {[Port:9999]}
- Job listing: 1. _pure-noise-phys-nnn_
               2. _gibbs-phys-nnn_
               3. _tta-phys-nnn_


> Baseline: </home/nfs/pedro/Baselines_NNN/>    {[Port:8888]}
- Job listing: 1. _pure-noise-base-nnn_
               2. _gibbs-base-nnn_
               3. _tta-phys-nnn_
               4. _baseline-no-unc-nnn_
               5. _physics-no-unc-nnn_

> Other
- Potentially relevant: Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty
                        See: https://arxiv.org/pdf/2006.06015.pdf

- Future plans (on job completion): 1. Dice analysis: Write script for this!
                                    2. Uncertainty maps: Entropy/ Variance
                                    3. Volumes graph
                                    3. Volumetric calibrations





# Log: 15.06.20
Aims: - Job checkup
      - Preliminary inference + dice analysis
      - DGX1 transfers


http://Job_Monitoring
> Physics: </home/nfs/pedro/Physics_NNN/>       {[Port:9999]}
- Job listing: 1. _pure-noise-phys-nnn_
                  Iters: 148,000                 {Converged}
               2. _gibbs-phys-nnn_
                  Iters: 101,000                 {Converged} Feature loss still decreasing
               3. _tta-phys-nnn_
                  Iters: 130,000                 {Converged}


http://Job_Monitoring
> Baseline: </home/nfs/pedro/Baselines_NNN/>    {[Port:8888]}
- Job listing: 1. _pure-noise-base-nnn_
                  Iters: 114,000                {Converged-ish}
               2. _gibbs-base-nnn_
                  Iters: 52,000                  [Continue]
               3. _tta-base-nnn_
                  Iters: 50,000                 {Converged-ish}
               4. _baseline-no-unc-nnn_
                  Iters: N/A                       _WIP_
               5. _physics-no-unc-nnn_
                  Iters: N/A                       _WIP_


> Notes
- No stratification loss for _pure-noise-base-nnn_: * Edited application file + restarted job from last iter
- Downloaded some inferences for _tta-base-nnn_: * Look a bit off from label
                                                 * Training w/ NO dropout, running inference, then comparing
                                                   * ~ 10k iterations should be enough? *Start at 53k and end at 60k?*

- Had to make new dataset split file: * <tilde/stratified_TTA_OOD_script_inferences.sh>
                                      * <tilde/SS_LowTD_MPRAGE_OOD_All_subjects/Restricted_30/inference_augmented_stratification_BS2_resampled/test.csv>


> OOD
- Directory reminder: </nfs/project/pborges/SS_LowTD_MPRAGE_OOD_All_subjects/Restricted_30> *Images*
-                     </nfs/project/pborges/LowTD_MPRAGE_corrected_params_All_subjects/Restricted_30> *Parameters*
                      </nfs/project/pborges/Labels_LowTD_MPRAGE_OOD_All_subjects/Restricted_30> *Labels*

- Dataset split file: <tilde/OOD_dataset_split.csv>              *Restricted: 837*
                      <tilde/OOD_Restricted_30_dataset_split.csv> *Restricted_30: 810*


> Other
...





# Log: 16.06.20
Aims: ...


> Dice calculations
- Preliminary inference comparison: * Ran inference for 0.5 Keep Prob TTA: </nfs/home/pedro/Baselines_NNN/TTA>
                                        0.87 averaged dice
                                    * Ran inference for 1.0 Keep Prob TTA: </nfs/home/pedro/Baselines_NNN/TTA_No_Do>
                                        0.85 averaged dice
                                    * Ran inference for 0.5 Keep Prob TTA Physics: </nfs/home/pedro/Physics_NNN/TTA>
                                        0.83 averaged dice: Respectable

- Don't forget to check on TTA noise job: _noise-tta-base-nnn_ </nfs/home/pedro/Baselines_NNN/TTA_noise>
- Don't forget to investigate better way to obtain dropout inferences faster





# Log: 17.06.20
Aims: - TTA noise job
      - Multi-dropout
      - Inference checking + set pure-noise-phys into inference

# Jorge meeting
- Useful MONAI functionality: https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FProject-MONAI%2FMONAI%2Fblob%2Fmaster%2Fexamples%2Fnotebooks%2Fpersistent_dataset_speed.ipynb&data=01%7C01%7Cm.jorge.cardoso%40kcl.ac.uk%7C546206e0e80f40c4a8e708d8120b5c3b%7C8370cf1416f34c16b83c724071654356%7C0&sdata=TnQ8sh%2FYNaLnKh4TsaSelK8rW%2FYOAVAOLr%2BQTpxdJ90%3D&reserved=0
  - Caches augmentations: Significantly reduces training time
  - Worth considering for TTA + if decide that want to simulate on demand

> Jobs
- Downloading preliminary results to analyse
  - Should be able to get some graphs done *Old scripts should still work just fine: Adjust to number of images + subjects*

> Multi-dropout
- Seems to have worked!
- Process: 1. Set a number of dropout samples, MC *As many as fits in GPU*
           2. Loop through MC times, calling self.net every time
           3. store segmentations in (X, Y, Z, MC) array

- Practical application: 1. Found GPU memory was limiting: Can't store 20 desired samples
                           2. For baseline TTA can store 10 *Run twice*
                              Running under: </nfs/home/pedro/Baselines_NNN/TTA/multi_test>
                                             </NiftyNet-5/niftynet/contrib/harmonisation/multi_dropout.py>
                           3. For physics TTA can't store 10, therefore store 7 *Run thrice*
                              Running under: </nfs/home/pedro/Physics_NNN/TTA/Inference>
                                             </nfs/home/pedro/NiftyNet-5/niftynet/contrib/harmonisation/stratified_physics_segmentation_application.py>

> Inferences
Expected outputs for each job: 1. [PureNoise]     180 images,  standard size          {BaselinePhysics}
                               2. [Gibbs]         180 images,  standard size x 4      {BaselinePhysics}
                               3. [TTA]           3600 images, standard size x MC     {None}
                               4. [BaselineNoUnc] 180 images,  standard size          {None}
                               5. [PhysicsNoUnc]  180 images,  standard size          {None}

- Can start processing both [Gibbs]
- Carry out reshaping on [PureNoise] baseline
  - **NOTE** Speckle patterns in background: * Persists in variance image!
                                             * Could be due to TRADITIONAL dropout usage
                                             * Also, high GM values in ventricle regions
                                             * Physics?: **NOT** present! Segmentations look good *Based on one example*
  - Fixed! Was running inference at latest iteration, which was 2000, NOT 100,000  #SPECKLE
    - Re-running inference
    - A bit of a setback given how long inferences take


> /storage/UNSURE_physics/Gibbs/inf_3: Many, many tests!





# Log: Week of 22.06.20  [TEST]
Aims: - Log all gibbs jobs
      - Discuss preliminary results

> Gibbs jobs
- Created a series of Gibbs jobs to compare relative performances
  - Largely to investigate effects of stratification changes
  - As a minor point, to also look into effects of keep_prob variations
  - All results processed + visualised using the standard script (with some alterations)
    </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/MPRAGE_inference_validation_paper_MiddleStandardisation.py>

- Job listing: * Gibbs:                   0.05 strat, 0.5 kp *Standard*
               * Gibbs_no_strat           0.00 strat, 0.5 kp
               * Gibbs_08                 0.50 strat, 0.8 kp
               * Gibbs_full_strat         1.00 strat, 0.5 kp
               * Gibbs_08_fuller_strat   10.00 strat, 0.8 kp
               * Gibbs_08_fullest_strat:  5.00 strat, 0.8 kp strat *I know, name compared to fuller is nonsensical*

- Consistency/ dice results: See: </home/pedro/Gibbs_comparison_plots_many.png>
                                  </home/pedro/Zeroed_Gibbs_comparison_plots_many.png>
  - Summary: * Baseline does surprisingly well with Dice + consistency isn't awful
             * Fullest strat has greatest consistency: Dice score is middling, however (~ 2 points lower than Baseline)
               * Training further as of writing: Both fuller + fullest
                                                 Saw that 20k steps made difference, might as well go for 20k more

- </nfs/home/pedro/Physics_NNN/Gibbs_08_2_strat/> still running
- </nfs/home/pedro/Physics_NNN/Gibbs_08_fullest_strat> 164k iters: Run inference!
  - Running inference at 164k

> Pure Noise jobs
- Baseline: * Variance looks good qualitatively, but extremely extremely low quantitatively
              * Sigma somewhat nonsensical, non-zero everywhere except in vicinity of active class
                * Makes sense? Where active class logit is low then sigma is unimportant: Assigned to random values
                               Where active class logit is high then sigma is important and is assigned to low value accordingly  ##SIGMARATIONALE


**[COVID]**
[runai submit tester-val -i 10.202.67.201:32581/pedro:covid -g 1 -p pedro --command python3 --args='/nfs/home/pedro/COVID/richard_code/train.py' -v /nfs:/nfs --run-as-user --host-ipc] *Safe-keeping*

> Starting off
- Main working directory: </nfs/home/pedro/COVID>
- OG Files contained in: </nfs/project/covid/CXR/KCH_CXR>
- OG Labels contained in: </nfs/home/pedro/COVID/Labels/KCH_CXR_Originals.csv>  *Saved from xlsx: Hopefully properly formatted*

- Created new docker image: [10.202.67.201:32581/pedro:covid]
  - Created a subdirectory for this: </nfs/home/pedro/DockerFiles/COVID>
  - A few notes: * apt-get commands in Dockerfile not amenable for some reason
                 * No requirements.txt, just a collection of common libraries
                   * Mainly: efficientnet-pytorch, torchvision, torch

- Cloned Richard's github repo containing preliminary scripts/ csv files: https://github.com/ricshaw/COVID_XRAY
- Refer to training script: <train.py>
  - A few minor issues: * Permission errors, fixed with --host-ipc flag
                          * See: https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990 *ipc-flag fix*

> Data specifics
- Labels: * 3436 entries
          * Age, sex, modality, symptom onset, death date (if applicable)
          * Some subject duplicates: Due to multiple scans: 1727 unique IDs
          * /nfs/project/covid/CXR/KCH_CXR

- Images: * 3437 DICOM files
          * 2627 unique entries
https://stackoverflow.com/questions/39880627/in-pandas-how-to-delete-rows-from-a-data-frame-based-on-another-data-frame *Exclusions*


> Other
- [pathlib] library: Excellent for applying various functions/ searches to filepaths
                      E.g.: Looping through files with certain extension (recursive): pthlib.Path(<DIR>).rglob('\*.ext') *Equivalent to glob but recursive default*
- [pandas] library: ...
                    Indexing: https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
                    Visualisation: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
                    Number of uniques across dataframe: <DATAFRAME>.nunique()
                    Accessing dataframe columns: <DATAFRAME>.<COLUMNNAME>
                    Accessing specific row: <DATAFRAME>.iloc[LOCATION]
                    Checking for nulls: <DATAFRAME>.isnull().any()
                    etc.





# Week of of 28.06.20
Aims: - COVID work
      - Final read of SASHIMI paper: https://www.overleaf.com/project/5e6e44e72178db00011b3670
      -

> Datasets
- Have issues relating to insufficient data: * After removing corrupted data, labeless data, only have 1400 relevant images
                                             * With addition of new dataset (covid-19-chest-x-ray-dataset) have ~2000 images
                                             * Have public datasets available: ChestXray-NIHCC


> covid-19-chest-x-ray-dataset
- Wrote script to extract relevant information from jsons: </data/COVID/dicom_to_jpg_pedro.py>
  - Json files not organised, so have to search for keywords
  - A lot of files have partial information: Filled in with "Missing"
    - Saved new csv: </data/COVID/covid-19-chest-x-ray-dataset_labels.csv>


> CHestXRay-NIHCC dataset
- Over 100000 images from ~30000 subjects
- See: https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
- Multiclass in PyTorch: https://discuss.pytorch.org/t/is-there-an-example-for-multi-class-multilabel-classification-in-pytorch/53579/6
  - Use BCELoss (Proper handling of OHE labels)


> Interpretability
- General guide: https://towardsdatascience.com/guide-to-interpretable-machine-learning-d40e8a64b6cf
- Saliency maps: Gradient of output scores wrt image
  - Implementation: https://medium.com/datadriveninvestor/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
- CAPTUM: PyTorch interpretabiliity library
          See: https://github.com/pytorch/captum
  **NOTE**: For multi-class see: https://github.com/pytorch/captum/issues/171

> k-fold training
- Average AUC: https://stats.stackexchange.com/questions/386326/appropriate-way-to-get-cross-validated-auc


> Notes
- Unique entries of list of dictionaries: * Convert dictionaries to strings, then run [set()]
                                          * set([str(a) for a in myList])
- Extend method returns 'None': Therefore don't use for assignment! *Use listA + listB instead*
- Keeping leading zeros in excel file: * [df.Filename = df.Filename.apply('="{}"'.format)] *Tells excel to treat the entry as formula that returns the quoted text*
                                       * See: https://stackoverflow.com/questions/41240535/how-can-i-keep-leading-zeros-in-a-column-when-i-export-to-csv
- Tensorboard in pytorch: https://pytorch.org/docs/stable/tensorboard.html
- ROC AUC vs PR: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
- Sigmoid vs Softmax for classification: https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/
  - **NOTE** Use softmax for MUTUALLY EXCLUSIVE labels, sigmoid otherwise
             Multi-class focal loss: See: https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289 *TLDR: Applicable*
                                     Also see: https://github.com/zhezh/focalloss *Implementation*
             Classification in general (OHE vs not): https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/


> Focal loss
- Facebook version seems to work the best for multiclass!: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

> Plots
- ROC-AUC plotter: </data/COVID/pedro_code/roc_auc_plotter.py>

> Pretraining
- Load weights: 1. Pre-training task has 14 classes: This is a mismatch!
                2. Therefore change final FCL to be able to accomodate this *temporarily: We do only want 4 target classes for the main task*
                  2a. [model.module.net._fc = nn.Linear(in_features=1536, out_features=14, bias=True)]
                  2b.                              <LoadThePretrainedModel>
                  2c. [model.module.net._fc = nn.Linear(in_features=1536, out_features=4, bias=True)]
- Freeze weights: * This depends on two factors: 1. Ratio of new data to pretrained data *Smaller ratio ==> More freezing*
                                                 2. Similarity of task *More similarity ==> More freezing*
                  * In our case, 1. is <LOW> and 2. is __MEDIUM__
                    * Start by freezing half the weights:
                    # Freeze some weights
                    <!-- frozen_dudes = [f'.{str(i)}.' for i in list(range(13))]
                    for name, param in model.module.net.named_parameters():
                      if any(frozen_dude in param for frozen_dude in frozen_dudes):
                          param.required_grad = False -->
- Ran: /nfs/home/pedro/COVID/pedro_code/death_time_classification_b3_focal_og.py
       COnverged very quickly


> Pandas
- Count uniques: * df = df.groupby('<DOMAIN>')[<ANCHOR>].nunique()
                   * Even better: df.fold.value_counts()[<SPECIFIC ENTRY>]
                 * See: https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups


**[RESULTS]**
Preliminary: DT-bo:       AUC = 0.61, acc = 0.86  {Overall}   </nfs/home/pedro/COVID/pedro_code/death_time_classification.py>
             DT-b3-focal  AUC = 0.70, acc = 0.84  {Overall}   </nfs/home/pedro/COVID/pedro_code/death_time_classification_b3_focal.py>
             DT-b7-focal  AUC = 0.78, acc = 0.84  {Overall}   </nfs/home/pedro/COVID/pedro_code/death_time_classification_b7_focal.py>

death_time_classification_b7_focal has latest changes! * Proper handling of Precision Recall
                                                       * Individual scores for EACH class
                                                       * Other minor changes
 **HOWEVER** pretraining.py contains captrum code *untested*

- Overlays (transparency): https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
- ROC-AUC plotting: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html  #ROC


> Interpretability: Occlusion specific:
- Occludes parts of the image in strides and sees how prediction changes
- See: https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html

- <death-time-b6-focal-softmax>: Actually b7!
- Running NON-OHE version of <death-time-b6-focal-softmax>: <death-time-b7-focal-standard>
  - Due to occlusions looking too similar: Should vary according to target, no?


> temp
pytorch_submitter restricted-covid-test-multi covid /nfs/home/pedro/COVID/pedro_code/pedro_train.py "/nfs/home/pedro/COVID/pedro_images/ /nfs/home/pedro/COVID/Data/covid-19-chest-x-ray-dataset/images" "/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG_restricted.csv /nfs/home/pedro/COVID/Labels/covid-19-chest-x-ray-dataset_labels.csv"

pytorch_submitter pretraining-b7 covid /nfs/home/pedro/COVID/pedro_code/pretraining_dgx1a_b7.py /nfs/project/covid/CXR/public_datasets/ChestXray-NIHCC/images/ /nfs/project/covid/CXR/public_datasets/ChestXray-NIHCC/Data_Entry_2017.csv

Histogram creation file (Kerstin): /home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/concise_histogram_analysis.py!
Permissions granting: [chmod a+rwx]

Atelectasis
Cardiomegaly
Consolidation
Edema
Effusion
Emphysema
Fibrosis
Hernia
Infiltration
Mass
Nodule
Pleural_Thickening
Pneumonia
Pneumothorax


> Other (Project)
- Deleting </storage/UNSURE_baselines/PureNoise/17_Jun> Contains unwanted speckle images (See ##SPECKLE)
- Deleting </storage/UNSURE_baselines/PureNoise/old> Contains unwanted "Noise" images *Due to nans in input images* (See ##NOISEIMAGES)
- Deleting </storage/UNSURE_baselines/PureNoise/reshaped> Contains "empty" volumes
- Deleting </storage/UNSURE_baselines/PureNoise/<NIIs>> Unsure, likely on cluster anyway


> Commands
Find files based on time of creation: find ./ -type f -mtime 2
find . -maxdepth 1 -type d -mtime 0 -exec cp -r {} /data/COVID/Data/KCH_news \;


> Folds creation
histogram_plots.py contains code to create "Time_To_Death" csv and save it


> SubNautica
DXVK_HUD=1 PROTON_LOG=1 %command%


> Papers
Unachievable Region in Precision-Recall Space
and Its Effect on Empirical Evaluation: https://icml.cc/2012/papers/349.pdf


# Log: New Week
- Aims: COVID


> Developments
- New folds script: ...
- Pretraining: * Seems significantly better with change to pos_weighting: [sqrt(ZerosInBatchLabels / OnesInBatchLabels)]
               * Job in question: <pretraining-b3-occ-sparse-sqrt>
               * BUT overfitting is notable after ~40k *Decreasing PRs, ROCs, validation loss*
                 * One epoch ~ 3500 iters, therefore 13 epochs into training
                 * If pretraining, start from there *i.e. 12 due to zero indexing*
- New pretraining: * From <pretraining-b3-occ-sparse-sqrt>, [Epoch12]
                   * Job name: <death-time-b3-folds-tta-pretrained-12>
                     *Rememeber, have to manually transfer wanted saved model into model directory beforehand*

- Micro vs Macro averaging: https://scikit-learn.org/0.18/modules/model_evaluation.html#average
                            Also: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
  - Gist: Micro average take ALL contributions from ALL classes, therefore classes are NOT weighted with respect to their size
          This means that you get a more accurate OVERALL view of your outputs, especially in a skewed setting
          When macro-ing, all classes are treated equally, which means that a minority class doing badly can skew results
            OTOH if you want to highlight minority class performance this is preferable!

**NOTE** Check AccessionID match BEFORE adding to filename (valid_dicoms) list!!
  Otherwise causes mismatch in sizes between filenames and scan times == disastrous (because the try block is exited after the check, so valid_dicoms only is upd.)


# Jorge meeting:
- COVID things to do: * Confusion matrix: Difficult to ascertain what classes are mistaken for which
                                          Allows ability to see this
                                          Specifically, want to see confusion between 48H and 1 week+/ Survival class
                                          Because ROC-AUC isn't end all be all, especially for skewed classes!
                        * See some examples of confusion: Actual images + occlusion
                      * Adding age, gender, other features into network
                      * Pretraining: Freeze everything and unfreeze from end in cascading manner
                                      E.g.: Every epoch unfreeze another layer

- Optimisation: See: https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d
                  Also: https://www.kaggle.com/ipythonx/tf-keras-melanoma-classification-starter-tabnet

- Model loading: Problem with pretraining (dictionary entry names mismatch) presists even if loading models from current training!
                   - Copied key adjusting loading: (alpha[2-4] scripts contain correction, **ONLY**)
        # Adjust key names
        keys_list = checkpoint['model_state_dict'].keys()
        new_dict = checkpoint['model_state_dict'].copy()
        for name in keys_list:
            new_dict[name[7:]] = checkpoint['model_state_dict'][name]
            del new_dict[name]
        model.load_state_dict(new_dict)
        # Main model variables
        # model.load_state_dict(checkpoint['model_state_dict'])

- For ease of comparison in tensorboard of major jobs: ranger/fold_0|full/fold_0|ranger

> In depth feature analysis

- Features only Death prediction: </data/COVID/pedro_code/logistic_full_bloods_ranger.py>
- ALL numeric features being employed in experiments: *See temp_bloods variable in /data/COVID/pedro_code/logistic_full_bloods_ranger.py*
                                      ['documentoutput_doc_dob', '.cLac', '.pCO2', '.pH', '.pO2', 'ALT',
                                       'Albumin', 'Alkaline Phosphatase', 'Aspartate Transaminase',
                                       'Bilirubin (Total)', 'Biochemistry (Glucose)', 'C-reactive Protein',
                                       'Creatinine', 'Estimated GFR', 'Ferritin', 'FiO2', 'Globulin', 'Glu1',
                                       'INR', 'Lymphocytes', 'NT-proBNP', 'Neutrophils', 'PCV', 'PLT',
                                       'Sodium', 'Troponin T..', 'Urea', 'WBC', 'cHCO3', 'D-Dimer',
                                       'NT-pro-BNP', 'CRP.', 'Lactate Dehydrogenase', 'Lactate Dehydrogenase.',
                                       'Diastolic BP', 'GCS Score', 'Heart Rate', 'NEWS2', 'Oxygen Saturation',
                                       'Respiration Rate', 'Systolic BP', 'Temperature', 'OBS Height',
                                       'OBS Weight', 'OBS BMI Calculation', 'Fever (finding)',
                                       'Cough (finding)', 'Fatigue (finding)', 'Dyspnea (finding)',
                                       'Muscle pain (finding)', 'Pain in throat (finding)',
                                       'Headache (finding)', 'Shivering or rigors (finding)',
                                       'Nausea (finding)', 'Loss of appetite (finding)', 'Diarrhea (finding)',
                                       'Abdominal pain (finding)', 'Nasal congestion (finding)',
                                       'Finding of sense of smell (finding)', 'Chest pain (finding)',
                                       'Atrial fibrillation (disorder)', 'Congestive heart failure (disorder)',
                                       'Finding of mental alertness (finding)',
                                       'Hypertensive disorder, systemic arterial (disorder)',
                                       'Diabetes mellitus (disorder)', 'Heart disease (disorder)',
                                       'Disorder of liver (disorder)', 'Chronic lung disease (disorder)',
                                       'Chronic obstructive lung disease (disorder)',
                                       'Interstitial lung disease (disorder)', 'Asthma (disorder)',
                                       'Cystic fibrosis (disorder)', 'Chronic kidney disease (disorder)',
                                       'Disorder of endocrine system (disorder)',
                                       'Neoplasm and/or hamartoma (morphologic abnormality)',
                                       'Cerebrovascular disease (disorder)',
                                       'Immunodeficiency disorder (disorder)']
- Pearson correlation script: </data/COVID/pedro_code/feature_correlation.py>
  - Results for </data/COVID/Labels/CXRs_latest_440_folds.csv> *Median imputation*

> Website
- Have folder: </home/pedro/my_blog>
  - Inside, run [bundle exec jekyll serve --trace]
  - Jekyll hosts temp version of website: Excellent for viewing changes! *Using VSCode to edit*


  labels = pd.read_csv('/data/COVID/Labels/CXRs_latest_440_folds.csv')


> Bloods only:
- Overfitting happens very quickly: ~3/4k iterations
  - Take models at ~4.8k iterations: Need to work out corresponding epoch
                                     14k iters == 700 epochs
                                     Therefore 4.8k iters ~ 150 epochs for some
                                     Should keep counter of "best" epoch per fold

> Registration procedure! </data/COVID/Data/KCH_CXR/registration/files>
- FInd and move dicoms: [find $PWD -name '*dcm' -exec mv -t $PWD {} + ]
- /home/pedro/dcm2niix/dcm2niix/build/dcm2niix/build/bin/dcm2niix <DIRECTORY> *Converts dicoms to niftiis within same folder*
  - Had to downsample (2048 cap on reg_aladin): fslmaths <FILE> -subsamp2 <OUTPUT>
- Registration: [reg_aladin -ref <FILE> -flo <FILE2> -ln 10 -lp 10 -maxit 20]
- Subtraction: [fslmaths <FILE> -sub <FILE2> <OUTPUT>]
- Remove nans: [seg_maths <FILE> -removenan <OUTPUT>] *Should be standard, replaces with zeros*
- Normalisation: [seg_maths <FILE> -sub $(fslstats <FILE> -m) -div $(fslstats <FILE> -s) <OUTPUT>] *Not sure about this*

# New files
dcm_nii_converter.sh [subject_num, status, acc1, acc2] *Run first*
nii_register.sh [fixed_nii, moving_nii] *Change into new subject directory, then run this*

> TODO
- Bloods attributions not showing correctly, have duplicate of image occlusion (See grid creation part of code: Parasitic!)
- Look into: <pretraining-b3-occ-sparse-sqrt>: Running for over 20 days!

- Example of bootstrapping job submission:
bootstrap_pytorch_submitter full-binary-cutmix-bs-occ covid /nfs/home/pedro/COVID/pedro_code/logistic_full_bloods_ranger_zero_impute_og_vitals_bootstrap.py /nfs/project/covid/CXR/KCH_CXR_JPG /nfs/home/pedro/COVID/Labels/cxr_news2_pseudonymised_filenames_latest_folds.csv train 79

> Pulmonary COVID project
- Much higher incidence of pulmonary embolism in COVID patients (3-6 times more)
- More likely to present to present to hospital with PE
  - Meaning: PE more likely to be COVID? Or going to hospital == increased PE risk?
- Are there specific COVID disease presentations which are correlated with PE?
  - Does lung damage correlate with PE?
  - Do changes in lung vessel diameter correlate with PE?
  - Risk factors for PE


bloods only script!: /nfs/home/pedro/COVID/pedro_code/logistic_full_bloods_ranger_zero_impute_og_vitals.py

# Updated logs: 15.08.20
Aims: - Partition regression analysis: Look into difference between attributions of correct and ver incorrect regressions
        - Also: Consider different approach to logging tabular attributions: Instead of reductive argmax, take continuous values for ALL variables
        - Imaging: Contrast has been adjusted thanks to information from David and code from Richard: Shouldn't require script changes!

> Logging blood attributions: Change code
- Currently just taking argmax of full attributions array, finding corresponding variable and logging that
  - Reductive: Does nothing to capture other variables' contribution to decision
               Better approach is to simply store all array values in list as an additional column
                During analysis, sum these and plot bar chart, as before *Just need to store mapping somehow*
                **NOTE** Essential to compare these results to previous ones: Might validate/ invalidate them!
                         Maybe start on regression analysis since that takes precedence now?




# Log: 01.09.20
Aims: - Clean up regression insight covid work (Pre-lunch)
      - Continue debugging code port (Inconsistent patch sizes? Image and Label)

# COVID Regression insight/ analysis
- Recap: * Designed network that regresses inverse time to death + 1, **NO IMAGING** *To account for people that died on same day of scan: Infinite!*
         * Split results between ICU and non-ICU: No immediately visible differences
         * Very variable results, even amongst patients that survived *i.e.: Label = 0*
           * See figures: [Imaging-reg-Non-ICU.png] etc. in </data/COVID/Figures>

         * Wanted to analyse differences: * Mainly good predictions vs bad predictions
                                          * Wrote preliminary script for this: </data/COVID/pedro_code/regression_analysis.py>
                                          * For now plots feature importances (based on occlusion) according to cutoffs
                                            * Current cutoffs: 500, 1000, 1579
                                            * More or less correspond to: Good prediction (survival), bad prediction (Survival), All predictions (Non-survival)

> Other recaps
  - Changed </data/COVID/pedro_code/reg_adaptive_logistic_full_bloods_ranger_zero_impute_og_vitals.py>
    - Now ALSO stores continuous features
    - What are these? Take exact attributions and store them
      - No argmaxing, get to look at how ALL features contribute towards decision making, NOT just most valuable one
      - Also gives extra insight due to negative attributions, i.e. those features that are DETRIMENTAL towards decision making
  - Already on cluster, no ned to upload
  - **CORRECTLY** run jobs that encompass changes: <reg-adaptive-no-imaging-cont-feat2>  *Models already in models folder*

TODO: * Additionally split between OVERALL good predictions and OVERALL bad predictions
        * Currently just using list sorts for this: Might be tricky?
        * Can still keep the same methodology: Order by regression difference then sort?
      * Split the good predictions (Sub-500) further: Any significant difference in feature importance observed?
      * General analysis: The why of the differences observed

[Done] Just a matter of applying sorting based on regression differences to all other lists (labels, predictions, continuous features)
       Results validate previous observations: Survival patients do not have very strong positive attributions, mostly all (weakly) negative
                                               Opposite for non-survival patients: Many more, strong, positive attributions
       **Reasoning** For survival patients, due to their majority, network "defaults" to these values
                     Therefore features have less of an importance in deciding outcome
                     For non-survival patients, due to their minority and because values differ greatly (not all zeros) features have greater influence
                     Therefore see more features with highly positive attributions

TODO: Should consider different attribution calculation methods: https://captum.ai/tutorials/House_Prices_Regression_Interpret
        Site links to regression example: No occlusion used!
          Attributions vary, sometimes significantly, depending on method used
          Baseline employed (All zero, or something else) can have significant effects
      <WorthConsidering> Integrated gradients, DeepLift, FeatureAblation
                         *Maybe should have been using feature ablation from the start for bloods...* https://captum.ai/api/feature_ablation.html


# PyTorch code port: Debugging
- Latest issue: Inconsistent patch sizes detected when looping through train loader
  - Potential solutions: 1. Change torchio code to allow functionality
                            - Seems to have the most potential
                            - Can start by eliminating error condition check: </home/pedro/.local/lib/python3.6/site-packages/torchio/data/subject.py> Line 130
                              - Checks to see if all arrays to be "patched" have same shape by looking at number of unique shapes
                              - If > 1 then raises error, changed this number to 2 so input and label can have different shapes
                              - [DONE] Seems to have worked! Output produced by _visualise_batch_patches_ shows matching across the board
                                - Run over 5 times with consistent results! Slight edits to the function to show ALL images in the batch + corresponding labels
                         2. Split label into two, pass each as separate variable before joining again
                            - Sounds good, but tricky; label is read directly from path taken from csv, hard to split in pipeline
                            - <NotWorth>
  - Other: Addressed issue with mismatched label/ output: Had to change final channel number in nnUNet! *One to Two*
           Regulatory ratio not used: Find use or exclude from training loop

# Log: 02.09.20
Aims: - Address remaining bugs with ported code
      - Verfiy training csv shuffling works as expected (batch subject homogeneity, non-consecutive subject batches etc.)

# Pytorch code debugging + further testing
- Addressed minor error with logging of "accuracy" (Not a relevant metric for segmentation!)
- Addressed minor error with stratification checker (classic case of reversed +ve/ -ve sign)
- **GLITCH** Have duplicated batches during training!
             Duplication number == num_workers parameter in DataLoader *e.g.: setting to 7 leads to every batch passed 7 times consecutively during training*
              Setting num_workers = 1 "solves" this at great speed expense
            [DONE] Found that torchio queue class has own num_workers parameter: Set here instead! https://torchio.readthedocs.io/data/patch_training.html#queue
                   Leave DataLoader parameter unset
                   Training now progresses as expected, intra-batch homogeneity preserved as verified by <stratification_checker>
                    *Aside: Check number of cpu threads available by importing multiprocessing and running: [multiprocessing.cpu_count()]*




> Other/ useful commands
Restart gnome shell: [gnome-shell --replace &]  https://askubuntu.com/questions/100226/how-to-restart-gnome-shell-from-command-line
                     [gnome-shell --version] *Find gnome shell version*

Kill tasks taking up GPU memory when nvidia-smi fails: [sudo fuser -v /dev/nvidia*] THEN kill all python3.6 processes
  See: https://stackoverflow.com/questions/63703615/execute-command-on-all-pids-outputted-by-primary-command





# Log: 03.09.20
Aims: - PyTorch code work
      - CXR covid meeting: Considerations

# PyTorch code
- Need to figure out how to properly evaluate validation! If stratification then need bs to be 4 as well?
  - Check what was done in NiftyNet? *Unfortunately cluster down for now*

> Other
- Internet issues...





# Log: 04.09.20
Aims: - COVID work: Longitudinal network
      - PyTorch code

# COVID work
- Want to create longitudinal network to try to eke out extra performance from images
  - A few different approaches: * Fixed time difference: 48H images
                                * Variable time difference: Any difference allowed, pass time difference as additional tabular feature (days?)
- Other TODO: * Investigate other means of interpretability relating to features *See log: 01.09.20*

> Longitudinal network
- Details: * Pass images (early + later) through network separately
           * BUT ensure layers are shared when doing so
           * Ensure same subjects stay in same fold

           * Starting point: </data/COVID/pedro_code/reg_imaging_adaptive_full_bloods_ranger_zero_impute_og_vitals.py>
             * Latest version of file that also includes imaging
             * Regression, but that should be fine to start with

           * Loading: Maybe start with unique subject csv, once subject is chosen pick two random images?

Also trying to figure out what the best way to select subjects in the dataloader is, since it isn't as straightforward as just looping through indices anymore: Maybe starting with the unique subject csv, and for the subject found choose two images from the full, all images csv?





# Log: 07.09.20
Aims: - PyTorch Project work! (only!)

# PyTorch Project
- Validation check!
  - Batch size 4 or one?: Go for 4 for now, easier to adapt to existing network/ other code

- Physics functionality
  - From NiftyNet: Post reduce size: [4, 2] *MPRAGE job*
  - Have to specify input channels for PyTorch FCLs
    - Calculate physics in network or not?
    - Best to not: Network should remain as "rigid" as possible, variations should occur in training script


> Crop concat outputs
- Non-physics:
torch.Size([4, 240, 10, 10, 10]) torch.Size([4, 240, 10, 10, 10])
torch.Size([4, 120, 20, 20, 20]) torch.Size([4, 120, 20, 20, 20])
torch.Size([4, 60, 40, 40, 40]) torch.Size([4, 60, 40, 40, 40])
torch.Size([4, 30, 80, 80, 80]) torch.Size([4, 30, 80, 80, 80])

- Physics
torch.Size([4, 240, 10, 10, 20]) torch.Size([4, 240, 10, 10, 20])
torch.Size([4, 120, 20, 20, 40]) torch.Size([4, 120, 20, 20, 40])
torch.Size([4, 60, 40, 40, 80]) torch.Size([4, 60, 40, 40, 80])
torch.Size([4, 30, 80, 80, 160]) torch.Size([4, 30, 80, 80, 80])

Why size mismatch?? Should there be a difference at all?
  **FIXED** Array channels in PyTorch are the second (i.e.: dim=1) dimension
            Therefore need to change tiling to reflect this: Tile across all dimensions BUT omit first (batch) and second (channel) instead of last
            Everything now works as expected!
            Ran local physics experiment (epochs=1) and everything ran to the end, fold model loading also seems to work





# Log: 08.09.20
Aims: - Move experiments over to DGX1a
        - Run tests: Physics + otherwise
      - Start adding in uncertainty losses + investigate validation

> Experiment planning
- Main experiments: * Baseline
                    * Physics (No stratification)
                    * Physics (Stratification: L2)
                    * Physics (Stratification: KLD)
- A few different stratification levels: 0.05 *default*, 0.1, 0.5

> KLD implementation: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
  - Careful with mean/ batchmean, etc.
  - In theory should work the same

> Data matters
  - Currently transferring data folders to dgx1a: </data/Resampled_Data/Images/SS_GM_Images> + </data/Resampled_Data/Labels/GM_Labels> to </nfs/project/pborges/Resampled_Data>
    - Labels have been merged in such a way as to have soft GM + background labels: [181, 217, 181, 2]
    - Facilitates loading process
  - MONAI data caching: https://docs.monai.io/en/latest/data.html#persistentdataset @PersistentDataset
    - First pass through transformation is applied, then saved to specified cache directory
    - Would involve adapting MONAI framework
    - Image generation quick, so probably no need: Test this out and investigate!

# Log: 09.09.20
Aims: - PyTorch experiments: 1. Transfer code + run experiments
      - ...
      - ...

> PyTorch experiments
- Transferred code to </nfs/home/pedro/PhysicsPyTorch>
  - Also added torchio as subdirectory: * Have edited torchio to allow for stratification functionality
                                        * Therefore can't simply transfer only code + pip3 install torchio
                                        * Added as <porchio>: Do not want NameSpace clashes!

- **FINALLY** got import of custom torchio to work properly! 1. [sys.path.append(DIRECTORY)] *ANY modules contained in this directory are visible now*
                                                             2. [import porchio] *porchio visible if directory that contains it is appended to sys path*
                                                             3. Had to therefore change all references from torchio to porchio

- Created new submission script: <physics_submitter>: Slight modification to <pytorch_submitter>
                                 Takes in relevant variables for physics/ baseline training *Allowing for easy modification!*

> Copied notes from laptop
- Created cluster version of training script: </home/pedro/PhysicsPyTorch/basic_train.py>
- Transferred sub-project folder to cluster: </nfs/home/pedro/PhysicsPyTorch>





# Log: 10.09.20
Aims: - Job checking
        - 2 x physics + 2 x baseline
      - Timing functionality: https://discuss.pytorch.org/t/loss-computation-time-constantly-increases/18906
      - OOD images: Find + csv (Probably won;t be able to get round to it today but let's see)

> Job Checking
- Training very unstable, in general: Base + phys
- initial suspicion: RangerLars optimizer
  - Main difference between current and old training scheme
  - Revert to Adam + LR scheduling: [phys-standard-test-adam-N] N = {112, 128, 144} *Patch size upper limit experiments*
                                    [base-standard-test]
    - Keeping in mind that there will be greater limitations when uncertainty branches are added
- Also implemented KLD as alternative to L2 between features *Should be employed with uncertainty, but not for now* [phys-kld-test-adam-144]
  - Pass as new argument for --experiment_mode *['standard', 'stratification', 'kld']*

> Other
- Time per iteration now logged (QoL)
- Fixed minor issues with validation variable naming
- WiFi issues: * "Forgot" 3STGS, seems to have helped
               * Turned off pwoer management for internal wifi card: https://unix.stackexchange.com/questions/269661/how-to-turn-off-wireless-power-management-permanently
                                                                     https://easylinuxtipsproject.blogspot.com/p/internet.html#ID2.1 (Other potential solutions)
                 * Will only come into effect after reboot, so be aware *Reboot s probably warranted at this point!*
- Potentially useful command to deal with Atom crashes: [atom --clear-window-state]  ##atom




# Log:  11.09.20
Aims: - Look into changing early stopping condition
        - Look into what it's even doing now!
      - COVID: Investigate new csvs: </nfs/project/covid/CXR/GSTT> <labs.csv> + <data.csv>
               Transferred them locally: </data/COVID/GSTT>

# PyTorch code
- Added feature loss tracking: Even for non-stratification experiments to serve as comparison
  - Also, currently not making use of regulatory ratio OR strengthening stratification over time
- Fixed early stopping: Was tracking dice LOSS, not dice SCORE
  - Therefore early stopping condition could never be met *Was checking for greater values when dice loss was decreasing*
    - Could also consider just using val loss as early stopping condition *Both are tracked: Interesting to compare*
  - Re-running kld job: Serves as test for new features

> Job notes
- Some difference depending on patch size: * All seem to be training well, PS 80 maybe an exception *Though PS 80 output val images still look good!*
                                           * PS 144 most stable, no outliers/ instabilities

> Pure noise uncertainty
- Seems to be best method to create upper + lower volume bounds
- Can extract directly from produced stochastic outputs: N passes produces N outputs
                                                         From N outputs obtain upper and lower bounds
                                                         Correspond to upper and lower bounds of aleatoric uncertainty
  - What about dropout? Fix dropout for every time N aleatoric samples are taken
    - Do this M times: Will have average variability due to aleatoric samples and average variability due to dropout
    - Sounds like a good plan in theory
      - KLD was somehow related to this? Need to refresh knowledge about this uncertainty method #TODO


# COVID GSTT data
- Finally!
- First impressions: [labs.csv] * Many entries per subject (10++ in some cases)
                                * Directly matching variables: [WBC, PLT, O2Sat, Ferritin, Ferritin COVID, D-Dimers, ESR]


> Other
- AI in medical imaging guidelines: https://www.nature.com/articles/s41591-020-1034-x.pdf





# Log: 14.09.20
Aims: - COVID prep for meeting with Finola
        - Compile list of questions
      - Project: Monitor jobs
        - Evaluate early stopping

# COVID
- Compiled list of questions: 1. Meaning of specimen dates: COVID tests more than one: In case of no admissions then arrives again
                                - Already a hospital patient keep evaluating
                                - Use first positive test to look for entries in system
                                  - Iterate: Look for data
                              2. Ferritin vs Ferritin COVID-19
                              3. eNot vs Sym
                              4. Columns names matching
                              5. Why difference between labs and data csvs?


# Project
- Jobs progressing along nicely (144s at least)
- BUT early stopping not triggered
  - Likely due to fluctuations resetting condition: Even if overall trend is improving
  - Consider using custom library, e.g.: https://github.com/Bjarten/early-stopping-pytorch
  - Same procedure as custom TorchIo to use? *i.e: git clone to repo?*

> Aside
- Maybe worth trying to calculate CoV within batches? (At val time)
  - Allows comparison even at train time in tensorboard
  - Easy to implement: 1. Turn off shuffling in val_loader
                       2. Calculate CoV per batch
                       3. Aggregate and average and log
                         3b. Change csv writing to reflect this: Maybe no need to re-calculate?

**NOTE** Val loss being calculated INCORRECTLY for stratification
         Assumes same batch behaviour as during training: Not the case!
          Val images are shuffled, so batches not from same subject
          Therefore feature loss is meaningless at val time
          Val time CoVs fixes this by also NOT randomising the val images
          Investigate if this is truly happening by printing val images before and after change to shuffling is made


Extra notes
- Remember to transfer new csr_fold_latest.csv to cluster *It's in the Labels folder*
- Code to create it in gstt_val.py (At bottom, probably should move or add flag)




# Log: 21.09.20
- Aims: - COVID: Interpretability
                 Single image: Latest and All comparison
        - Project: Job monitoring
                   Look at intermediate csvs and plot classic CoV plots: Compare
                   Don't necessarily need all folds to have finished training
                   OOD set up: Inference only

> Other
- scp vs rsync: https://stackoverflow.com/questions/20244585/how-does-scp-differ-from-rsync
  - Use rsync with -P option for large files: Call again if interrupted!
  -



# COVID asides
- I think there may be a way to eke out a few extra val images, but I'm unsure on implementation: Basically you can have a case where you have two X-rays for example, and the patient has, say, two time points in the csv. Right now if the diff...
- ...


# COVID summary of past week
- Implementation of interpretability: * Had to jump through a few hoops to deal with Captum setup
                                        * E.g.: model wrappers in order to ignore parts of model inputs


# Network stuff
https://askubuntu.com/questions/168032/how-to-disable-built-in-wifi-and-use-only-usb-wifi-card
Add the following line to /etc/network/interfaces:

iface wlan0 inet manual

NetworManager doesn't manage interfaces configured in the interfaces file. Replace wlan0 with the interface you want to disable, if it's not the name of the built-in interface.

Then restart network manager

sudo service network-manager restart


# Project: Jorge meeting
- Bizarre stratified behaviour: * Calculate volume for each image multiple times
                                  * There should be NO variation: Investigates presence of stochasticity
                                * Look at images: Loading proper data types, look at actual images to see if there are any obvious differences
- Linear behaviour: * Average across subjects: In theory (With inclusion of error bars) linear behaviour should disappear
- Compare against baselines, obviously
- OOD experiments validation for generalisability

# Captum
Captum doubts: https://github.com/pytorch/captum/issues/475
1 or 2 output nodes? https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons

# Stratification problems
- Problems thought isolated to stratification are actually present in normal physics experiments!
  - <csv_processor.py> on: * [Stratification] </home/pedro/PhysicsPyTorch/logger/new_tests_physics/phys-stratification-test-adam-144-es-covdice-5000>
                           * [Baseline]       </home/pedro/PhysicsPyTorch/logger/base-standard-test-adam-144-es-covdice>
                           * [Physics]        </home/pedro/PhysicsPyTorch/logger/phys-standard-test-adam-144-es-covdice>
- Problematically, baseline performs BETTER than physics!
  - But, when layered behaviour is addressed this may change





# OOD work
- Relevant directories: Images: </nfs/project/pborges/SS_LowTD_MPRAGE_OOD_All_subjects/Restricted_30>
                        Labels: </nfs/project/pborges/Labels_LowTD_MPRAGE_OOD_All_subjects/Restricted_30>
                                </data/OOD_Labels> *Local!*
                        Physics: </nfs/project/pborges/LowTD_MPRAGE_corrected_params_All_subjects/Restricted_30> *Unnecessary, probably*

- Downloading images (for csv creation) to: </data/Resampled_Data/Images/OOD>
  - Can now create csv using </home/pedro/PhysicsPyTorch/fold_csv_creator.py>
  - Need to integrate OOD into script: * Inference time only?
                                       * Incorporate into validation?
                                         * Should be possible, but then is there a point to current validation?
                                         * Should copy fold behaviour from current csv to OOD csv?

- Inference: Run all models on inference set? Should think so
  - Therefore, for each fold, find latest file


Old question about renaming files: https://stackoverflow.com/questions/63633967/removing-changing-pattern-from-filenames-in-directory-in-linux
  Basically, to get rid of [00001 - 32000] from files for TTA inference
  *Not sure if a script was written for this?? Recall doing something with this before*
  Created </bin/label_trimmer.sh> 0 to 3500: Should cover everything

Temperature scaling! [Deep Generative Model for Synthetic-CT Generation with Uncertainty Predictions]
  For regression sigmas directly translate into uncertainty bounds

# Inference functionality
- Using grid_sampling method from TorchIO: https://torchio.readthedocs.io/data/patch_inference.html?highlight=gridsample
  - Basically, create a [BespokeDataset] instance of inference subjects
  - Loop through samples, pick image, which are sampled with grid sampler
  - **NOTE** patch_overlap variable does not seem to work as expected, only works when set to zero...

# GradCAM
- See GradCAM plus plus


# Quick summary of new week
- Compared complete baseline + baseline 50 to other experiments
  - Strat 50 Physics seems to still do best! Great!
  - Not so obvious looking at averages, but dice still higher, as well

- Started trying to use monai for inference: https://docs.monai.io/en/latest/inferers.html
  - Seems to work well! Except overlap doesn't seem to do anything...
  - Running different inferences at different overlaps: So far nothing
  - Actually, there seems to be a difference: * But (obviously) results in large computation times
                                              * 0.75 far too large a number
                                              * Probably go for default: 0.25

# Uncertainty
- Found out that network has one fewer conv layers after physics concatenation
  - Definitely worth experimenting with addition of another layer (as with NiftyNet experiments)
    - Probably create separate model + training file for this

# RunAI things
- Potential for significant speed increases using Horovod: https://docs.run.ai/Researcher/Walkthroughs/walkthrough-distributed-training/#introduction
  - Requires building of new docker image (fine) and slightly changing submission (fine) AND changes to model parallelism (maybe not fine!) @Walter
  - Changes to model parallelism: See chat with Walter: * https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py
                                                          * Parameter broadcasting, data partitioning, optimizer wrapping
  - Probably worth trying, could save SIGNIFICANT amounts of time (< 50% performance improvements)
  - In case this is unsuccessful: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html (Distributed data parallel)
> Docker Image deletion: [docker image rm <ImageID>] *Get IDs from docker images command*

# COVID
- Re-running all images model: Making sure to delete model and clear cache after every fold!
  - Also with some changes to training (See Richard's message)

# Log: 14.10.20
Aims: - Uncertainty work: Implementation + running preliminary experiment
      - COVID: Re-run missing experiments for table
        - All + latest images (Multi), All images (singular timepoint)
          - Singular timepoint: Re-done with deleted models every epoch: Worse results on KCH but GSTT HO make more sense

Zach paper reminder: Towards safe deep learning: accurately quantifying biomarker uncertainty in neural network predictions
                     https://arxiv.org/pdf/1806.08640.pdf


# Log: 15.10.20
Aims: - Dropout implementation + preliminary experiments
      - COVID: Sort out occlusion attributions (At least the coding)
        - COVID meeting notes + work to follow up on
      - Investigate performance of heteroscedastic experiments

> Profiling
- Profiling library: https://github.com/pyutils/line_profiler
  - Script for this that takes in python file: </bin/profiler>  *local*
  - Running on <tilde/tester.py>  *corrected paper function for unc*

# Check subset only having a single timepoint
# Timepoint at admisssion: Secondary objective


# Log: 16.10.20
Aims: - Dropout experiments
      - COVID: Image retrieval + Upload
               Send David info. about cluster access

# Project: Uncertainty
- Added another flag: [dropout_level] to scripts + submission file
  - 0.0 - 1.0, self explanatory
  - Running experiments, all identified by <dropout> in name

**NOTE** Running into frequent memory issues even when running with BS 4, 128 PS
         Runs just fine for a few hundred iterations then crashes
         Also, hetero. unc. VERY time consuming, largely because of normal_dist_array creation
          *Line profiler indicated > 50% of time spent on array creation alone!*
         Look into VERY useful guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
         Also look into: https://medium.com/@raghadalghonaim/memory-leakage-with-pytorch-23f15203faa4
         **SOLUTION** Create tensor DIRECTLY on GPU instead of creating then passing to GPU
                      i.e. [torch.normal(..., device=torch.device('cuda:0'))]
                           <NOT>   [torch.normal(...).cuda()]
                      Speed-up: (5 + 2)s -> (1 + 2)s!
- Job running under new changes: <phys-unc-strat-50-dropout-cuda-direct>
                                 <phys-unc-strat-50-cuda-direct>
- Monitor VERY closely!

# COVID
- Made some modifications to <gstt_val.py> to allow for earliest functionality
  - Modified <pytorch_submitter> latest_flag: 'latest', 'earliest', 'all' *'all' default*
  - Ran earliest experiments (single TP): See: https://docs.google.com/presentation/d/1i18O9jaXwAvIm6-6sqHfOQfEG3G1FUrU-KwIfe9z9UI/edit#slide=id.g99523900c3_3_0
    - Very similar to all timepoints
- Other findings in terms of data findings: * ~50% of subjects with only ONE entry for KCH
                                            * ~98% of subjects with only ONE entry for GSTT!
                                            * Seems to render inference on GSTT pointless for longitudinal networks...

> Data retrieval
- 86 primary objective images: </nfs/project/covid/CXR/GSTT/primary_obj_imgs>




# Log: Week of 19.10.20
Aims: - Requested Occlusion/ Saliency plots: Coding
      - Uncertainty jobs: Investigate stability + segmentation quality

# COVID
- Coded some changes to <gstt_val.py>: 1. Created entirely separate occlusion section below training loop, above GSTT val loop
                                       2. Load in best epoch, run occlusion
                                       3. Also load in preds.csv to ascertain model predictions
                                       4. Also save saliency in addition to occlusion
  - Interpretability only performed on those Ascension numbers that match those sent by James *100 images*

- Need to train a few models: * Images only (Done)
                              * Images + demographics
                              * Images + demographics + bloods

> Demographics
- Comprises: Age, Gender, Ethnicity, BMI
  - Create flag? Probably best approach
  - Created flag: [--demo_flag] False/ True
    *14 flags, getting a bit messy*

- Altered [get_feats] + [get_feats_noi]
  - Checks for state of demo_flag: If True then changes to bloods to JUST include BMI
- Also changed temp_bloods in similar manner: Include print statment *Should be 5 columns when demo flag is on*
- Network now gets number of inputs directly from number of columns present in temp_bloods


# Project
- Summary of optimisations: 1. Direct creation of variables on GPU <SIGNIFICANT>
                            2. Deletion of variables every loop *Model: Every fold, arrays: Every iteration/ epoch for val*
                               2a. Particularly important after validation!
                                   Otherwise have validation arrays stored during next training loop! <SIGNIFICANT>
                            3. Set parameters to None instead of optimizer.zero_grad() [MINOR]
                            4+. Model parallelism, avoid variable calls, variable pre-allocation [MINOR]

- Job statuses: - All hetero. uncertainty jobs exhibiting **SIGNIFICANTLY** lower val dice scores: ~ 0.75
                  - Slight exception to 100 passes job: <phys-unc-strat-50-dropout-cuda-direct-100>: ~ 0.85 *Still increasing*
                - Dropout exclusive job <phys-strat-50-dropout> performing slightly worse, but within reason (0.5 dropout)
  - Making maximum number of epochs 60 instead of 30: Seems like networks had not yet converged @ 30

> Blurred images: Bridging project
- Had to dig up some old info!
- <smoother.sh> Runs [seg_maths -smo <STD>] on all images in a directory with a specified <STD>
                Smooths images without downsampling
                Currently running on </nfs/project/pborges/Resampled_Data/SS_GM_Images> -> </nfs/project/pborges/Resampled_Data/Smoothed_SS_GM_Images>
                  Fairly quick process, 1/sec
- Previous work involved some amount of noise + bias augmentation: Currently only have bias augmentation
  - Validation will tell if this is necessary

- Training scheme: 1. Train network with blurred images
                   2. Ascertain performance through validation: DIrectly compare against equivalent for non-blurred
                   3. Ascertain performance on SABRE *Dig up metrics for old experiments if possible!*

Potentially useful tensorboard regex: phys-unc-strat-50-cuda-direct|phys-stratification-test-adam-144-es-covdice-50-v2/fold_0|-100|phys-strat-50-dropout|03|physicser/fold_0


# Log: 21.10.20
Aims: - Continued uncertainty monitoring
      - COVID: Check occlusion jobs

# Project: Jorge meeting
- Paper/ Journal plan is as such: 1. Performance of stratification
                                    1a. Performance of uncertainty in addition to stratification: * Bland-Altman  plots
                                                                                                  * Error bars on consistency plots
                                    2a. How well it generalises to OOD
                                  2. Bridging study: Round two: Replicate previous results + uncertainty
                                  3. Apply to differen datasets: - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5302120/pdf/nihms826509.pdf
                                                                 - Demonstrate increased consistency across datasets given physics

- For point 3: Need to run SPGR experiments! Absolutely crucial!
               Generate SPGR images from scratch? Or generate during training?
               Latter would be amazing: Would only need to load qMPMs:
                Try this today!

- Misc. memory issues: https://stackoverflow.com/questions/64465757/how-to-optimise-memory-usage-of-looped-loss-function-in-pytorch

> On demand image generation
- Process: 1. Load in MPMs instead of realisations *i.e. MPM file for sub N instead of MPRAGE for sub N*
            1a. Create a new csv to reflect this: Have 27 MPMs + Labels [Multiply by] 120
           2. Add a new augmentation/ pre-processing that generates image of protocol of choice *SPGR, MPRAGE, etc.*
             2a. Parameters are randomly selected from a list: Uniform random selection

- New files: [Script] </nfs/.../mpm_basic_train.py>
             [CSV]    </nfs/.../MPM_physics_csv_folds.csv>

- New directories: [Labels] </nfs/project/pborges/Resampled_Data/MPM_GM_Labels>
                   [MPMs]   </nfs/project/pborges/Resampled_Data/SS_Resampled_MPMs>
                            </home/pedro/Project/Simulations/AD_dataset/MPMs/resampled_MPMs/SS_resampled_MPMs> *Local! MPMs NOT GM labels!*


- Code additions: * MPRAGE "augmentation" </home/pedro/PhysicsPyTorch/porchio/transforms/augmentation/intensity/mprage_generation.py>
                  * MPM csv creator: </home/pedro/PhysicsPyTorch/mpm_fold_csv_creator.py>
                  * Bash files for skull stripping MPMs: </bin/4d_merger.sh> + </bin/mpm_masker.sh>

- Other changes needed for implementation: 1. Augmentation is cyclic: Loops through images and generates MPRAGE
                                           2. TIs are randomly selected from a given range *0.6 to 1.2 usually*
                                             2a. Then replace placeholders in <sample>
                                           3. Thankfully, patch selection behaviour seems to work with no modifications needed

- Jobs: <mprage-generation-test>: Seems to be running
        Need to monitor: * Patches: [Correct!]
                         * Physics: [Correct!]
                         * Batch homogeneity [Correct!]
  - **NOTE** Seem to have problem with ZNormalization (whitening): Images are NaNs when turned on...
    - Fixed this: Get rid of NaNs after image generation [tensor[t]]

> SPGR parameters
- Cater to dataset parameter choices
- Also see: http://mriquestions.com/spoiled-gre-parameters.html *T1 vs T2s vs PD weighting*
            https://www.radiologycafe.com/radiology-trainees/frcr-physics-notes/t1-t2-and-pd-weighted-imaging

> Useful TB regex: phys-unc-strat-50-cuda|phys-unc-strat-10|phys-unc-strat-50-dropout-cuda-direct/|mprage





# Log: 26.10.20
Aims: - Investigate dropout + unc. inference
        - Especially unc: Far trickier to accomplish
      - Miscellaneous COVID work: Make sure all occlusions are finished!

Dropout related: https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738
Hetero related: https://stackoverflow.com/questions/63934070/gradient-accumulation-in-an-rnn *Potential solution for more inference passes*

> Hetero. uncertainty
- Problems: Output ROIs not complete, print some variable sizes to check:

Non-uncertainty: torch.Size([1, 2, 181, 217, 181])
                 (slice(21, 181, None),
                 slice(57, 217, None),
                 slice(21, 181, None))
                 torch.Size([160, 160, 160])
                 torch.Size([1, 2, 160, 160, 160])

Uncertainty: torch.Size([1, 2, 181, 217, 181])
             (slice(0, 160, None),
             slice(0, 160, None),
             slice(0, 160, None))
             torch.Size([160, 160, 160])
             torch.Size([1, 2, 160, 160, 160])


> Sliding window inference: Explained
1. For each ROI: Predict segmentation and/ or unc
2. Store these values in respective lists
3. Join them together in SEPARATE loop

For uncertainty, either: 1. For each ROI calculate seg and unc N times
                          1a. THEN loop and put everything together
                         2. For each inf_pass calculate all ROIS, put together
                          2a. Then loop through all ROIS *This one is favoured!* <- Went for this!

> Test output: Hetero. uncertainty
- Seems to work, have 5D output x 2 as expected
- **BUT** Uncertainty is zero: * Probably related to training rather than inference implementation
          Output is definitely not Softmaxed?? Need to check that softmax is being carried out across proper dimension

- **NOTE** Fixed! Was assigning softmaxed output to storage tensor, THEN overwriting it with stochastic tensor
            Therefore seemed like there was no Softmax AND seemed like stochastic tensor was empty *Was never writing to it in first place*

- Still have problems with heteroscedastic uncertainty being << epistemic uncertainty: Doesn't even alter graphs!
  - Still important to visualise: * At inference save SIGMA not stochastic logits
                                  * Or save BOTH
                                  * Also maybe keep track of magnitudes? i.e.: physics sigma vs base sigma vs phys sigma
                                    *Also see sigma rationale tag: ##*
                                    Look at thesis + paper! Rememeber that it had some insight into this subject ###TODO

> Dropout problems
- Output never seems to change at all: 1. Tried setting all dropout calls to .train(True): No success
                                       2. Tried to set all dropout calls to .train immediately prior to eval() call: No success
                                       3. Maybe have to call eval then set to train??
**NOTE** Silly mistake: * By definition model.eval() <DISABLES> dropout layers!
                        * Therefore do NOT call model.eval() when dropout is involved! *Maybe not even at all since there i no batch norm either*
                        * Now everything seems to work!


# COVID
Interesting occlusion subject: [RJZ74071735]
Quick email draft: - Email James with all occlusion results
                   - Slides: https://docs.google.com/presentation/d/17_-TkZrpoYzoTc5flQ2q_uia-VOgP2axXJBSy29pmZ8/edit#slide=id.p





# Log: 27 - 29.10.20
Aims: - Dropout inferences
        - Histogram processing scripts
        - Other miscellaneous uncertainty scripts
      - COVID: On hold until Friday

> Test-time analyses
- Decided to limit OOD to 2000ms: Too unreasonable looking segmentations beyond this point: No point
  - Have local copy of OOD "Labels": </data/OOD_Labels/Individual>
  -

> Uncertainty: Dropout
- Saw before that while hetero. uncertainty is <<< epistemic, maps look reasonable *See: ...*

- First proper inference directory: </nfs/home/pedro/PhysicsPyTorch/logger/Figures/ood-phys-strat-50-dropout>
  - General organisational structure: * N Inference{n} folders: Each corresponds to a different MC sample for the OOD dataset
                                      * Files are labelled accordingly:
                                      * <dropout_organiser.sh> [Directory] [Num_samples - 1] creates this structure

  - Downloaded locally for analysis: </home/pedro/PhysicsPyTorch/logger/ood-phys-strat-50-dropout>
    **NOTE** Want to be able to perform analyses remotely moving forward
    Adapted a few scripts for this purpose: </home/pedro/Project/ImageSegmentation/CNR_SNR/Scripts/pytorch_uncertainty_analyses.py> *Local*
                                            => </nfs/home/pedro/PhysicsPyTorch/Analyses/pytorch_uncertainty_analyses.py> *Remote*

  - Analysis script: 1. Iteratively loops through subjects
                     2. For each subject loop through all the MC directories
                     3. Calculate mean, STD, error rate, min volume, max volume
                     4. Mean + vols combined and stored in <Combined_seg_uncertainty/inf_{n}> to mirror non-dropout exps *Facilitaties normal analyses*
                     5. STD + error rate combined and stored in <Qualitative_uncertainty>

  - Histograms: 1. Addition to <inf_processor.py>
                2. Read relevant Qual. vols for each subject
                3. Use np.histogram2d() to organise data

[Experiments] <ood-phys-strat-50-dropout> * Bland Altman plot as expected: Increasing uncertainty with increasing error
                                            * But uncertainty seems somewhat overestimated: More dropout samples preferable?
                                          * Volume consistency: Seems reasonable, but has clearly deteriorated

              <base-mprage-generation-test-stand-dropout> * SE: Baseline generation experiment with 0.5 dropout
              <mprage-generation-test-dropout> * SE: Physics generation experiment with 0.5 dropout
              <phys-unc-strat-50-dropout-cuda-direct> * Physics non-gen. with hetero. with 0.5 dropout

Useful regex: mprage-generation-test|phys-unc-strat-50-cuda-direct-proper/|phys-unc-strat-50-dropout-cuda-direct/


> Asides
- Conditional layers for scanner generalisability: https://arxiv.org/pdf/1709.07871.pdf *FiLM*



# Log: 03.11.20
Aims: - Organise and process dropout inferences
        - Analyses: * Changes to BA with more/ fewer dropout samples
                    * Changesto error bars with more/ fewer dropout samples
      - 1D histogram equalisation! Implement it!

> Dropout organisation and processing
- Continued developing <pytorch_uncertainty_analyses.py>  --input_directory --num_dropout_samples
- Edited <mpm_basic_train.py> to automatically create Inference{N} subdirectory structure

- Running dropout processing job: <dropout-processing-test> [ood-mprage-generation-test-dropout]


> 1D histogram equalisation: Lifted from @[Log 05.06.20]
- Wrote dedicated script: </home/pedro/Project/Simulations/Scripts/histogram_normalisation_personal.py>  *Bare-bones, will need to modify + test later*
- **Need** to write consolidated script that: 1. Reads inference images (N-samples per)
                                          2. Computes CDFs from percentiles *How?*
                                            2a. Percentiles across N-samples
                                            2b. Summation across percentiles to get percentile volumes
                                            2c. CumSum over volumes
                                            2d. Derive uniform distribution to match against {HowToDoThis?checkScriptForCurrentImplementation}
                                          3. Calculates 1D HN transform
                                          4. Plots comparison error bar plots: 4a. Mean volume against GT volume
                                                                               4b. Uncalibrated error bars
                                                                               4c. Calibrated error bars

- Started writing script for this: </home/pedro/Project/Simulations/Scripts/calibrated_volumetric_estimates.py>  ##CALIBRATION

- Moving forward: 1. Will probably need way to generate CDFs at validation time moving forward:
                    2. Each fold automatically has a differen subject subset: No changes needed here
                  3. BUT generating N (Where N > 20) samples at validation time and creating CDFs from these might be cumbersome
                  4. Probably better to set it up as an "inference" except on val set instead of actual inference set
                    5. Maybe encode in OOD_flag: OOD, non-OOD, or val? If val carry out these val-time CDFs

> Save normal inferences as inf_{N}! Make sure to change BOTH mpm_basic_train AND basic_train!

Quick note for dmeographics jobs: Number of features is N + 1! Extra one due to relative time difference (First scan and current scan)
Take all arguments after the nth in bash: ${@:n+1} *Takes n+1th and all subsequent ones*
Also: ${@[N,-M]} *Take Nth to Mth arguments*





# Log: 06.11.20
Aims: - Test and run dgx_inf_processor on mprage generation jobs
      - Process graadcams for COVID work
        - Scaling, overlays, saving as pngs + npys
      - Send various emails: GradCAMs + GSTT data acquisition (Leo cc'd)

> COVID
- Processed GradCAMs: 1. Had to scale by factor of 1000 before plotting
                        1a. Values were of order of 1e-8: Rounded to zero when plotting
                      2. Otherwise same process as with occlusion, overlays and non-overlays saved
- Sent GradCAMs email: But not GSTT related email
  - James will look through results with Tina + Hasti this week and get back to us


# Log: 09.11.20 - 10.11.20
Aims: - Get back on track with project
        - Inferences with dgx_inf_processor *Were these already done?*
      - Look at </home/pedro/Project/Simulations/Scripts/calibrated_volumetric_estimates.py> script
      - Experiments: What actually still needs to be done? 1. Physics MPRAGE generation experiment looks "good enough" *With dropout*
                                                           2. Baseline MPRAGE generation experiment: Running uncertainty analysis -> vol consistency
                                                           3. Non-generation experiments: Need to be checked/ re-run potentially
                                                           4. SPGR experiments: Need to be checked

# Project
- Implement proper error bar handling *Older code should have this, should be a matter of simple adaptation*
  - IGNORE volumetric calibration for now
  - Always ends up taking an inordinate amount of time
  - Related, see: https://campus.datacamp.com/courses/introduction-to-data-visualisation-in-python/analyzing-time-series-and-images?ex=13

- Dropout analysis: <base-mprage-generation-test-stand-dropout>
                      Awful dropout results! Clearly network hasn't trained enough
                      Queued job for re-training

- Other jobs: <phys-unc-strat-50-cuda-direct-proper>
                Deleted to make room for mprage-generation hpo job
              <spgr-generation-test-1>
                Reduced stratification due to seemingly too large a constraint posed by having this value be == 50
              <hpo-mprage-generation-test-do>
                HPO experiment looking at: [0.1, 1, 10, 50, 100] stratification
                For physics, not baseline (yet!)
              **NOTE** A few extra flags are necessary: --parallelism <INT> *How many hpo jobs to run in parallel*
                                                        --completions <INT> *How many hpo jobs to run in total: Pick number of potential combinations*

> Bland-Altman plots
- Seem nonsensical: * Two "modes" observed instead of continuous line
                    * Odd behaviour with region of high uncertainty around
                    * Solutions: 1. Argmax when calculating error rates + std -> plot BA from there
                                 2. Investigate the odd region of medium uncertainty with zero error rates

# COVID
- Need to converge on a few models: * Definitive results for ALL table combinations
                                      * All demographics jobs
                                      * Proper longitudinal job handling *Waiting on Finola for this*

- Paper focus: Might be sufficient to show that bloods alone is worthwhile for predictions
- Interpretability: * Select meeting with James + other clinicians about interpretability + main results
                      * 18th or 20th
                    * Investigate positivity/ negativity of occlusion: 1. Post question on github
                                                                       2. Run with negative prediction


# Log: 11.11.20
Aims: - Bland-Altman invesitgations
        - New runai submission: Figure out potential bugs
      - HPO: Monitor

> Single loading of MPM generation jobs
- There is no point to loading N volumes per batch if the subject is the same
  - Rather, should load one volume, generate N samples, then pass these N samples as a batch
  - Do this via following changes: 1. Change batch size to one
                                   2. Alter generation augmentation layer: Take MPM and generate N volumes, assign that to the sample
                                     2a. Re-assign names in training loop

> HPO
- Error when yaml file tried to be written to at validation time: https://github.com/facebookresearch/DensePose/issues/37
  - Could be related to PyYaml version: Try creating docker image with version 3.12 instead of current 5.3
  - **NOTE** Error was associated with EXISTING runai.yaml files: They need to be deleted first!

> Bland Altman plots
- Altered <dgx_inf_processor.py> + <pytorch_uncertainty_analyses.py> to save + process argmaxed versions of qualitative volume (STD + Error rate)
  - Potential solution to current BA problem (1.)
  **NOTE** Current Argmax-BA plots very sparse
           Faster to analyse locally, downloading argmaxed Qual. volumes to investigate from <ood-phys-strat-50-dropout>





# Log: 12.11.20
Aims: - Start converging on final experiments
      - ...

Main final experiments: [Physics] Non-strat physics: *Current potentially good enough*
                                  Strat physics: *Current potentially good enough*
                                  Non-strat generative physics: *Maybe not needed?*
                                  Strat generative physics:         <single-load-mprage-gen-test-no-do>
                                  Strat generative physics dropout: <single-load-mprage-gen-test>

                        [Baseline] Non-strat baseline: *Current potentially good enough?*
                                   Strat baseline: *Current potentially good enough?*
                                   Non-strat generative baseline: <base-single-load-mprage-gen-test>
                                   Strat generative baseline:
                                   Strat generative 

                        [HPO] Physics stratification: [0.1, 1, 10, 50, 100] strat, [0.5] dropout:   <hpo-single-load-mprage-generation-test-do>

                        single-load-mprage-gen-test-no-do                   Running    1h    dgx2-a      10.202.67.201:32581/pedro:pytorch               Train  pedro    pedro  1 (1)                       1 (0)
                        base-single-load-mprage-gen-test-no-do              Running    1h    dgx2-a      10.202.67.201:32581/pedro:pytorch               Train  pedro    pedro  1 (1)                       1 (0)
                        base-single-load-mprage-gen-test                    Running    1h    dgx2-2      10.202.67.201:32581/pedro:pytorch               Train  pedro    pedro  1 (1)                       1 (0)
                        single-load-mprage-gen-test                         Running    1d    dgx1-1      10.202.67.201:32581/pedro:pytorch               Train  pedro    pedro  1 (1)                       1 (0)


> Benchmarks (Non-extensive augmentations)
Non-generative: [4.5-7s]
Generative (N load): [4.5-6s]
Generative (Single load): [2.8-4s] *Potentially up to 50% faster!*


> Aside
#
Relevant calibration papers: Deep Generative Model for Synthetic-CT Generation with Uncertainty Predictions (Simpler), Towards safe deep learning: accurately quantifying biomarker uncertainty in neural network predictions (Validation-time, Zach's paper)
#

**CATASTROPHY** Logs writeen while in Portugal have become corrupted
                Attempts to recover fruitless: Lost -> Corrupted
                Detailed EVERYTHING about porting code...
                Recovering what details I can: https://gist.github.com/Hsankesara/e3b064ff47d538052e059084b8d4df9f (Implementation of UNet)
                  Obviosuly, a lot of modifications necessary: Convert to nnUNet architecture, add physics, add unc. etc.
.
