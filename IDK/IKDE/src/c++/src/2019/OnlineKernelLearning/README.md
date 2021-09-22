Online Kernel Learning
======================

This software was obtained from the following website:

http://lsokl.stevenhoi.org/

The software was modified to add the Isolation Kernel compoment.

Additional Command line parameters
----------------------------------

Batch mode:
* -opt KERNEL-IK_OGD
* -ik_model <aNNE, iNNE, DotProduct> : DotProduct is for pre-generated data file.
* -ik_psi <value>
* -ik_sets <value>


Online mode:
* -ik_mode_online : to select online mode
* -ik_ol_init_bk_size : the initial training size
* -ik_ol_acc_bk : output the accuracy after 'x' number of blocks


Additional Notes
----------------

The original software is rather fragile. For example, if the number of data points within a data file is a multiple of 256, then the program could crash. I haven't attempted to supply any fixes to the original software.
