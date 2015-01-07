Documentation
====================

al
----

.. automodule:: al.learning_curve
   :members:

front_end.cl
------------

.. automodule:: front_end.cl.run_al_cl
   :members:

Examples
^^^^^^^^
   This code runs :mod:`front_end.cl.run_al_cl` with the following parameters:
   
   * number of trials - 5
   * strategy - rand
   * bootstrap - 10
   * budget - 500
   * step size - 10
   * subpool - 250
   * data paths - ../../../data/imdb-binary-pool-mindf5-ng11 ../../../data/imdb-binary-test-mindf5-ng11

   .. code-block:: python
		
      python run_al_cl.py -c MultinomialNB -nt 5 -st rand -bs 10 -b 500 -sz 10 -sp 250 -d ../../../data/imdb-binary-pool-mindf5-ng11 ../../../data/imdb-binary-test-mindf5-ng11

   The output of this code is:
   
   .. image:: _images/run1.png
      :width: 50%


utils.utils
-----------

.. automodule:: utils.utils
   :members:
