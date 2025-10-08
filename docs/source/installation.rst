.. _installation:

Installation and Example Training
===================================

This section provides instructions for downloading, configuring, and running the ML research platform on the Frontier system at Oak Ridge National Laboratory.

STEP 1: Downloading the ML Research Platform
---------------------------------------------

The ML research platform is a Python project managed with Git. To obtain the source code, clone the `repository <https://github.com/grnydawn/miles-credit.git>`_ and switch to the **frontier** branch using the following commands:

.. code-block:: bash

	git clone https@github.com:grnydawn/miles-credit.git
	cd miles-credit
	git checkout frontier

STEP 2: Creating a Python Virtual Environment and Installing Packages
----------------------------------------------------------------------

Run the following make command in the top-level directory of the repository:

.. code-block:: bash

	make venv
	make install

.. note::
   The make commands are provided for easy installation. Please refer to the **Makefile** for details about the commands.


STEP 3: Running an Example Model
---------------------------------

.. note::
   This section explains how to run the model on an interactive node rather than in batch mode.

First, obtain an allocation on an interactive Frontier computing node.

.. code-block:: bash

	salloc -A <ACCOUNT> -J inter -t 2:00:00 -q debug -N 1

Next, retrieve the name of the node using the **hostname** command.

.. code-block:: bash

    hostname

Finally, run the following commands to execute the CrossFormer model on the node.

.. code-block:: bash

    export MASTER_ADDR=<HOSTNAME>
    export MASTER_PORT=29500

    cd <CREDIT_REPO>

    make train_xformer_srun


.. note::
   * **<HOSTNAME>** is the same name displayed when running the **hostname** command above.
   * **<CREDIT_REPO>** is the top-level directory of the Git repository.
   * The **make** commands are provided for easy execution of training. Please refer to the **Makefile** for details about the commands.
   * The training configurations is specified in **<CREDIT_REPO>/config/frontier_xformer.yml**
   * The input data files used in the example training are located under **/lustre/orion/cli115/scratch/grnydawn/data/CREDIT**, which is a part of files provided from `https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview <https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview>`_

Once all the above steps are completed successfully, the training progress will be displayed on the screen with progress bar indicators.

