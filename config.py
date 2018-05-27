import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Data set locations
eur_usd_loc = "/run/media/two-a-day/Elements/academic/datasets/EURUSD-15m-2010-2016/EURUSD_15m_BID_01.01.2010-31.12.2016.csv"

# trained model locations
baseline_wights_loc = "bin/trained_models/baseline/weights.best.hdf5"
