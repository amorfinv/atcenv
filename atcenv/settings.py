'''Read in the configuration file'''
import os
import math
import atcenv.units as u

# get config file path
cfgfile = os.path.join('', 'settings.cfg')

exec(compile(open(cfgfile).read(), cfgfile, 'exec'), globals())
