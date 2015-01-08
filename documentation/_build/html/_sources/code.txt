Documentation
====================

al
----

.. automodule:: al.instance_strategies
   :members:

.. automodule:: al.learning_curve
   :members:

front_end.cl
------------

.. automodule:: front_end.cl.run_al_cl
   :members:

Examples
^^^^^^^^
   The following code runs :mod:`front_end.cl.run_al_cl` with the following parameters:

   * number of trials - 5
   * strategy - rand
   * bootstrap - 10
   * budget - 500
   * step size - 10
   * subpool - 250
   * data paths - ../../../data/imdb-binary-pool-mindf5-ng11 ../../../data/imdb-binary-test-mindf5-ng11

   .. code-block:: python

      python run_al_cl.py -c MultinomialNB -nt 5 -st rand -bs 10 -b 500 -sz 10 -sp 250 -d ../../../data/imdb-binary-pool-mindf5-ng11 ../../../data/imdb-binary-test-mindf5-ng11

   *The output of this code is:*

   *Status:*

   .. code-block:: python

      Loading took 17.88s.

      trial 0
      trial 1
      trial 2
      trial 3
      trial 4

   *Data output is placed in a file in your current working directory. The
   default filename is avg_results.txt.*

   *Sample Data Output:*

   .. code-block:: python

      rand
      accuracy
      train size,mean
      10,0.557016
      20,0.538432
      30,0.534664
      40,0.575320
      50,0.651672
      60,0.621416
      70,0.670400
      80,0.645680
      90,0.659520
      100,0.610160
      110,0.658024

   *Plot Image:*

   .. image:: _images/run1.png
      :width: 50%

utils.utils
-----------

.. automodule:: utils.utils
   :members:
