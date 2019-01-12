from __future__ import division
import iotbx.pdb
import mmtbx.utils
from cctbx import maptbx
import mmtbx.model
import mmtbx.f_model
from mmtbx import map_tools
from cctbx import miller
import mmtbx.refinement.real_space.individual_sites
from libtbx.utils import null_out
from scitbx.array_family import flex

pdb_str_answer = """\
CRYST1   43.974   32.795   33.672  90.00  90.00  90.00 P 1           0
ATOM      1  N   SER A   6      32.017  14.662  18.780  1.00 25.00           N
ATOM      2  CA  SER A   6      32.491  16.016  18.597  1.00 25.00           C
ATOM      3  C   SER A   6      31.281  16.942  18.673  1.00 25.00           C
ATOM      4  O   SER A   6      30.129  16.498  18.589  1.00 25.00           O
ATOM      5  CB  SER A   6      33.198  16.210  17.263  1.00 25.00           C
ATOM      6  OG  SER A   6      32.231  16.113  16.216  1.00 25.00           O
ATOM      7  N   LYS A   7      31.503  18.271  18.681  1.00 25.00           N
ATOM      8  CA  LYS A   7      30.496  19.340  18.601  1.00 25.00           C
ATOM      9  C   LYS A   7      29.677  19.217  17.315  1.00 25.00           C
ATOM     10  O   LYS A   7      28.446  19.222  17.395  1.00 25.00           O
ATOM     11  CB  LYS A   7      31.163  20.714  18.673  1.00 25.00           C
ATOM     12  CG  LYS A   7      31.803  21.026  20.017  1.00 25.00           C
ATOM     13  CD  LYS A   7      32.466  22.392  20.011  1.00 25.00           C
ATOM     14  CE  LYS A   7      33.084  22.713  21.362  1.00 25.00           C
ATOM     15  NZ  LYS A   7      33.760  24.040  21.363  1.00 25.00           N
ATOM     16  N   TYR A   8      30.355  19.097  16.187  1.00 25.00           N
ATOM     17  CA  TYR A   8      29.674  19.039  14.901  1.00 25.00           C
ATOM     18  C   TYR A   8      28.677  17.888  14.888  1.00 25.00           C
ATOM     19  O   TYR A   8      27.519  18.196  14.605  1.00 25.00           O
ATOM     20  CB  TYR A   8      30.679  18.886  13.759  1.00 25.00           C
ATOM     21  CG  TYR A   8      31.593  20.078  13.582  1.00 25.00           C
ATOM     22  CD1 TYR A   8      31.213  21.156  12.792  1.00 25.00           C
ATOM     23  CD2 TYR A   8      32.832  20.124  14.204  1.00 25.00           C
ATOM     24  CE1 TYR A   8      32.044  22.247  12.627  1.00 25.00           C
ATOM     25  CE2 TYR A   8      33.671  21.213  14.045  1.00 25.00           C
ATOM     26  CZ  TYR A   8      33.272  22.270  13.256  1.00 25.00           C
ATOM     27  OH  TYR A   8      34.103  23.355  13.092  1.00 25.00           O
ATOM     28  N   ALA A   9      29.154  16.816  15.478  1.00 25.00           N
ATOM     29  CA  ALA A   9      28.292  15.634  15.446  1.00 25.00           C
ATOM     30  C   ALA A   9      27.035  15.856  16.289  1.00 25.00           C
ATOM     31  O   ALA A   9      25.939  15.469  15.868  1.00 25.00           O
ATOM     32  CB  ALA A   9      29.071  14.407  15.895  1.00 25.00           C
ATOM     33  N   ARG A  10      27.185  16.460  17.483  1.00 25.00           N
ATOM     34  CA  ARG A  10      26.022  16.782  18.294  1.00 25.00           C
ATOM     35  C   ARG A  10      25.096  17.774  17.594  1.00 25.00           C
ATOM     36  O   ARG A  10      23.865  17.608  17.611  1.00 25.00           O
ATOM     37  CB  ARG A  10      26.464  17.332  19.652  1.00 25.00           C
ATOM     38  CG  ARG A  10      25.307  17.788  20.546  1.00 25.00           C
ATOM     39  CD  ARG A  10      24.394  16.657  21.045  1.00 25.00           C
ATOM     40  NE  ARG A  10      25.123  15.776  21.930  1.00 25.00           N
ATOM     41  CZ  ARG A  10      24.571  14.673  22.470  1.00 25.00           C
ATOM     42  NH1 ARG A  10      23.310  14.331  22.245  1.00 25.00           N
ATOM     43  NH2 ARG A  10      25.349  13.908  23.244  1.00 25.00           N
ATOM     44  N   SER A  11      25.656  18.830  16.999  1.00 25.00           N
ATOM     45  CA  SER A  11      24.796  19.797  16.328  1.00 25.00           C
ATOM     46  C   SER A  11      24.023  19.104  15.189  1.00 25.00           C
ATOM     47  O   SER A  11      22.817  19.363  14.996  1.00 25.00           O
ATOM     48  CB  SER A  11      25.633  20.949  15.775  1.00 25.00           C
ATOM     49  OG  SER A  11      26.258  21.618  16.858  1.00 25.00           O
ATOM     50  N   ASN A  12      24.703  18.182  14.492  1.00 25.00           N
ATOM     51  CA  ASN A  12      24.000  17.600  13.356  1.00 25.00           C
ATOM     52  C   ASN A  12      22.971  16.649  13.947  1.00 25.00           C
ATOM     53  O   ASN A  12      21.893  16.502  13.359  1.00 25.00           O
ATOM     54  CB  ASN A  12      24.999  16.892  12.442  1.00 25.00           C
ATOM     55  CG  ASN A  12      25.835  17.902  11.678  1.00 25.00           C
ATOM     56  OD1 ASN A  12      25.419  19.042  11.445  1.00 25.00           O
ATOM     57  ND2 ASN A  12      27.029  17.445  11.303  1.00 25.00           N
ATOM     58  N   PHE A  13      23.265  16.003  15.082  1.00 25.00           N
ATOM     59  CA  PHE A  13      22.318  15.118  15.756  1.00 25.00           C
ATOM     60  C   PHE A  13      20.988  15.813  16.041  1.00 25.00           C
ATOM     61  O   PHE A  13      19.923  15.259  15.779  1.00 25.00           O
ATOM     62  CB  PHE A  13      22.940  14.502  17.017  1.00 25.00           C
ATOM     63  CG  PHE A  13      22.133  13.341  17.572  1.00 25.00           C
ATOM     64  CD1 PHE A  13      21.179  13.659  18.526  1.00 25.00           C
ATOM     65  CD2 PHE A  13      22.310  12.015  17.225  1.00 25.00           C
ATOM     66  CE1 PHE A  13      20.414  12.685  19.146  1.00 25.00           C
ATOM     67  CE2 PHE A  13      21.547  11.035  17.836  1.00 25.00           C
ATOM     68  CZ  PHE A  13      20.594  11.360  18.784  1.00 25.00           C
ATOM     69  N   ASN A  14      21.213  16.988  16.628  1.00 25.00           N
ATOM     70  CA  ASN A  14      20.049  17.742  17.051  1.00 25.00           C
ATOM     71  C   ASN A  14      19.202  18.191  15.858  1.00 25.00           C
ATOM     72  O   ASN A  14      17.963  18.217  15.954  1.00 25.00           O
ATOM     73  CB  ASN A  14      20.454  18.918  17.917  1.00 25.00           C
ATOM     74  CG  ASN A  14      20.929  18.483  19.289  1.00 25.00           C
ATOM     75  OD1 ASN A  14      20.529  17.414  19.777  1.00 25.00           O
ATOM     76  ND2 ASN A  14      21.750  19.329  19.917  1.00 25.00           N
ATOM     77  N   VAL A  15      19.822  18.530  14.719  1.00 25.00           N
ATOM     78  CA  VAL A  15      19.030  18.830  13.529  1.00 25.00           C
ATOM     79  C   VAL A  15      18.307  17.578  13.023  1.00 25.00           C
ATOM     80  O   VAL A  15      17.126  17.621  12.675  1.00 25.00           O
ATOM     81  CB  VAL A  15      19.890  19.484  12.451  1.00 25.00           C
ATOM     82  CG1 VAL A  15      19.149  19.556  11.121  1.00 25.00           C
ATOM     83  CG2 VAL A  15      20.298  20.894  12.908  1.00 25.00           C
ATOM     84  N   CYS A  16      19.029  16.437  12.987  1.00 25.00           N
ATOM     85  CA  CYS A  16      18.416  15.187  12.566  1.00 25.00           C
ATOM     86  C   CYS A  16      17.168  14.854  13.388  1.00 25.00           C
ATOM     87  O   CYS A  16      16.179  14.317  12.867  1.00 25.00           O
ATOM     88  CB  CYS A  16      19.482  14.086  12.693  1.00 25.00           C
ATOM     89  SG  CYS A  16      18.928  12.419  12.300  1.00 25.00           S
ATOM     90  N   ARG A  17      17.201  15.184  14.681  1.00 25.00           N
ATOM     91  CA  ARG A  17      16.074  14.882  15.552  1.00 25.00           C
ATOM     92  C   ARG A  17      14.865  15.807  15.335  1.00 25.00           C
ATOM     93  O   ARG A  17      13.755  15.419  15.709  1.00 25.00           O
ATOM     94  CB  ARG A  17      16.517  14.939  17.019  1.00 25.00           C
ATOM     95  CG  ARG A  17      17.436  13.782  17.397  1.00 25.00           C
ATOM     96  CD  ARG A  17      16.687  12.477  17.616  1.00 25.00           C
ATOM     97  NE  ARG A  17      15.940  12.546  18.854  1.00 25.00           N
ATOM     98  CZ  ARG A  17      14.899  11.775  19.177  1.00 25.00           C
ATOM     99  NH1 ARG A  17      14.343  10.946  18.283  1.00 25.00           N
ATOM    100  NH2 ARG A  17      14.403  11.805  20.413  1.00 25.00           N
ATOM    101  N   TRP A  18      15.057  16.988  14.743  1.00 25.00           N
ATOM    102  CA  TRP A  18      13.949  17.932  14.642  1.00 25.00           C
ATOM    103  C   TRP A  18      12.692  17.345  13.998  1.00 25.00           C
ATOM    104  O   TRP A  18      11.593  17.573  14.522  1.00 25.00           O
ATOM    105  CB  TRP A  18      14.391  19.167  13.854  1.00 25.00           C
ATOM    106  CG  TRP A  18      13.301  20.175  13.658  1.00 25.00           C
ATOM    107  CD1 TRP A  18      12.603  20.420  12.510  1.00 25.00           C
ATOM    108  CD2 TRP A  18      12.779  21.077  14.641  1.00 25.00           C
ATOM    109  NE1 TRP A  18      11.683  21.418  12.718  1.00 25.00           N
ATOM    110  CE2 TRP A  18      11.768  21.838  14.018  1.00 25.00           C
ATOM    111  CE3 TRP A  18      13.063  21.322  15.989  1.00 25.00           C
ATOM    112  CZ2 TRP A  18      11.047  22.821  14.693  1.00 25.00           C
ATOM    113  CZ3 TRP A  18      12.344  22.297  16.657  1.00 25.00           C
ATOM    114  CH2 TRP A  18      11.349  23.034  16.009  1.00 25.00           C
ATOM    115  N   PRO A  19      12.771  16.574  12.899  1.00 25.00           N
ATOM    116  CA  PRO A  19      11.565  15.979  12.326  1.00 25.00           C
ATOM    117  C   PRO A  19      11.047  14.763  13.086  1.00 25.00           C
ATOM    118  O   PRO A  19      10.000  14.241  12.713  1.00 25.00           O
ATOM    119  CB  PRO A  19      11.964  15.551  10.908  1.00 25.00           C
ATOM    120  CG  PRO A  19      13.446  15.418  10.977  1.00 25.00           C
ATOM    121  CD  PRO A  19      13.912  16.436  11.965  1.00 25.00           C
ATOM    122  OXT PRO A  19      11.676  14.326  14.050  1.00 25.00           O
TER     123      PRO A  19
END
"""

pdb_str_poor = """\
CRYST1   43.974   32.795   33.672  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   6      31.724  14.653  18.584  1.00 25.00           N  
ATOM      2  CA  SER A   6      32.409  15.966  18.580  1.00 25.00           C  
ATOM      3  C   SER A   6      31.467  17.150  18.710  1.00 25.00           C  
ATOM      4  O   SER A   6      30.301  16.999  19.130  1.00 25.00           O  
ATOM      5  CB  SER A   6      33.257  16.092  17.294  1.00 25.00           C  
ATOM      6  OG  SER A   6      32.539  15.679  16.146  1.00 25.00           O  
ATOM      7  N   LYS A   7      31.918  18.343  18.260  1.00 25.00           N  
ATOM      8  CA  LYS A   7      31.061  19.550  18.399  1.00 25.00           C  
ATOM      9  C   LYS A   7      30.156  19.715  17.141  1.00 25.00           C  
ATOM     10  O   LYS A   7      29.143  20.448  17.195  1.00 25.00           O  
ATOM     11  CB  LYS A   7      31.927  20.786  18.657  1.00 25.00           C  
ATOM     12  CG  LYS A   7      31.096  21.939  19.154  1.00 25.00           C  
ATOM     13  CD  LYS A   7      31.962  23.123  19.646  1.00 25.00           C  
ATOM     14  CE  LYS A   7      31.476  23.579  20.993  1.00 25.00           C  
ATOM     15  NZ  LYS A   7      30.276  24.496  20.877  1.00 25.00           N  
ATOM     16  N   TYR A   8      30.469  18.994  16.063  1.00 25.00           N  
ATOM     17  CA  TYR A   8      29.572  18.984  14.937  1.00 25.00           C  
ATOM     18  C   TYR A   8      28.555  17.868  15.066  1.00 25.00           C  
ATOM     19  O   TYR A   8      27.374  18.073  14.777  1.00 25.00           O  
ATOM     20  CB  TYR A   8      30.337  18.863  13.629  1.00 25.00           C  
ATOM     21  CG  TYR A   8      31.080  20.121  13.214  1.00 25.00           C  
ATOM     22  CD1 TYR A   8      32.195  20.033  12.422  1.00 25.00           C  
ATOM     23  CD2 TYR A   8      30.660  21.403  13.654  1.00 25.00           C  
ATOM     24  CE1 TYR A   8      32.895  21.167  12.052  1.00 25.00           C  
ATOM     25  CE2 TYR A   8      31.342  22.495  13.282  1.00 25.00           C  
ATOM     26  CZ  TYR A   8      32.450  22.384  12.522  1.00 25.00           C  
ATOM     27  OH  TYR A   8      33.130  23.525  12.160  1.00 25.00           O  
ATOM     28  N   ALA A   9      28.994  16.725  15.554  1.00 25.00           N  
ATOM     29  CA  ALA A   9      28.109  15.559  15.628  1.00 25.00           C  
ATOM     30  C   ALA A   9      26.945  15.834  16.575  1.00 25.00           C  
ATOM     31  O   ALA A   9      25.796  15.520  16.231  1.00 25.00           O  
ATOM     32  CB  ALA A   9      28.845  14.297  16.076  1.00 25.00           C  
ATOM     33  N   ARG A  10      27.218  16.465  17.693  1.00 25.00           N  
ATOM     34  CA  ARG A  10      26.128  16.847  18.599  1.00 25.00           C  
ATOM     35  C   ARG A  10      25.227  17.900  17.919  1.00 25.00           C  
ATOM     36  O   ARG A  10      24.001  17.927  18.165  1.00 25.00           O  
ATOM     37  CB  ARG A  10      26.697  17.394  19.901  1.00 25.00           C  
ATOM     38  CG  ARG A  10      25.785  17.200  21.094  1.00 25.00           C  
ATOM     39  CD  ARG A  10      25.492  15.704  21.348  1.00 25.00           C  
ATOM     40  NE  ARG A  10      26.679  14.884  21.714  1.00 25.00           N  
ATOM     41  CZ  ARG A  10      26.626  13.587  22.029  1.00 25.00           C  
ATOM     42  NH1 ARG A  10      25.488  12.930  22.034  1.00 25.00           N  
ATOM     43  NH2 ARG A  10      27.749  12.915  22.348  1.00 25.00           N  
ATOM     44  N   SER A  11      25.785  18.649  17.019  1.00 25.00           N  
ATOM     45  CA  SER A  11      25.019  19.689  16.376  1.00 25.00           C  
ATOM     46  C   SER A  11      24.337  19.178  15.126  1.00 25.00           C  
ATOM     47  O   SER A  11      23.220  19.613  14.791  1.00 25.00           O  
ATOM     48  CB  SER A  11      25.957  20.841  16.086  1.00 25.00           C  
ATOM     49  OG  SER A  11      26.516  21.306  17.304  1.00 25.00           O  
ATOM     50  N   ASN A  12      24.989  18.204  14.406  1.00 25.00           N  
ATOM     51  CA  ASN A  12      24.339  17.527  13.295  1.00 25.00           C  
ATOM     52  C   ASN A  12      23.385  16.439  13.793  1.00 25.00           C  
ATOM     53  O   ASN A  12      22.876  15.667  12.960  1.00 25.00           O  
ATOM     54  CB  ASN A  12      25.367  16.884  12.373  1.00 25.00           C  
ATOM     55  CG  ASN A  12      26.068  17.899  11.495  1.00 25.00           C  
ATOM     56  OD1 ASN A  12      25.699  19.079  11.370  1.00 25.00           O  
ATOM     57  ND2 ASN A  12      27.120  17.438  10.801  1.00 25.00           N  
ATOM     58  N   PHE A  13      23.186  16.323  15.101  1.00 25.00           N  
ATOM     59  CA  PHE A  13      22.290  15.309  15.665  1.00 25.00           C  
ATOM     60  C   PHE A  13      20.992  15.953  16.076  1.00 25.00           C  
ATOM     61  O   PHE A  13      19.922  15.301  16.118  1.00 25.00           O  
ATOM     62  CB  PHE A  13      22.942  14.616  16.880  1.00 25.00           C  
ATOM     63  CG  PHE A  13      22.590  13.187  17.068  1.00 25.00           C  
ATOM     64  CD1 PHE A  13      21.440  12.801  17.717  1.00 25.00           C  
ATOM     65  CD2 PHE A  13      23.440  12.164  16.608  1.00 25.00           C  
ATOM     66  CE1 PHE A  13      21.139  11.456  17.906  1.00 25.00           C  
ATOM     67  CE2 PHE A  13      23.111  10.801  16.791  1.00 25.00           C  
ATOM     68  CZ  PHE A  13      21.953  10.487  17.441  1.00 25.00           C  
ATOM     69  N   ASN A  14      21.023  17.216  16.398  1.00 25.00           N  
ATOM     70  CA  ASN A  14      19.846  17.912  16.885  1.00 25.00           C  
ATOM     71  C   ASN A  14      18.804  18.178  15.785  1.00 25.00           C  
ATOM     72  O   ASN A  14      17.618  17.867  15.948  1.00 25.00           O  
ATOM     73  CB  ASN A  14      20.267  19.166  17.663  1.00 25.00           C  
ATOM     74  CG  ASN A  14      19.989  19.042  19.142  1.00 25.00           C  
ATOM     75  OD1 ASN A  14      19.799  17.942  19.652  1.00 25.00           O  
ATOM     76  ND2 ASN A  14      19.988  20.188  19.880  1.00 25.00           N  
ATOM     77  N   VAL A  15      19.264  18.711  14.623  1.00 25.00           N  
ATOM     78  CA  VAL A  15      18.309  18.899  13.541  1.00 25.00           C  
ATOM     79  C   VAL A  15      17.995  17.592  12.806  1.00 25.00           C  
ATOM     80  O   VAL A  15      17.305  17.587  11.781  1.00 25.00           O  
ATOM     81  CB  VAL A  15      18.805  19.995  12.569  1.00 25.00           C  
ATOM     82  CG1 VAL A  15      17.968  20.030  11.345  1.00 25.00           C  
ATOM     83  CG2 VAL A  15      18.946  21.356  13.282  1.00 25.00           C  
ATOM     84  N   CYS A  16      18.500  16.475  13.300  1.00 25.00           N  
ATOM     85  CA  CYS A  16      18.042  15.189  12.753  1.00 25.00           C  
ATOM     86  C   CYS A  16      16.978  14.577  13.644  1.00 25.00           C  
ATOM     87  O   CYS A  16      16.284  13.634  13.256  1.00 25.00           O  
ATOM     88  CB  CYS A  16      19.198  14.210  12.618  1.00 25.00           C  
ATOM     89  SG  CYS A  16      18.706  12.605  11.904  1.00 25.00           S  
ATOM     90  N   ARG A  17      16.920  15.048  14.899  1.00 25.00           N  
ATOM     91  CA  ARG A  17      15.792  14.728  15.755  1.00 25.00           C  
ATOM     92  C   ARG A  17      14.672  15.732  15.621  1.00 25.00           C  
ATOM     93  O   ARG A  17      13.660  15.592  16.328  1.00 25.00           O  
ATOM     94  CB  ARG A  17      16.194  14.677  17.218  1.00 25.00           C  
ATOM     95  CG  ARG A  17      17.440  13.963  17.488  1.00 25.00           C  
ATOM     96  CD  ARG A  17      17.296  12.444  17.201  1.00 25.00           C  
ATOM     97  NE  ARG A  17      15.965  11.977  17.707  1.00 25.00           N  
ATOM     98  CZ  ARG A  17      15.590  10.700  17.591  1.00 25.00           C  
ATOM     99  NH1 ARG A  17      16.447   9.834  17.176  1.00 25.00           N  
ATOM    100  NH2 ARG A  17      14.486  10.234  18.146  1.00 25.00           N  
ATOM    101  N   TRP A  18      14.835  16.748  14.777  1.00 25.00           N  
ATOM    102  CA  TRP A  18      13.761  17.735  14.605  1.00 25.00           C  
ATOM    103  C   TRP A  18      12.558  17.200  13.841  1.00 25.00           C  
ATOM    104  O   TRP A  18      11.423  17.467  14.274  1.00 25.00           O  
ATOM    105  CB  TRP A  18      14.348  19.012  13.948  1.00 25.00           C  
ATOM    106  CG  TRP A  18      13.584  20.260  14.263  1.00 25.00           C  
ATOM    107  CD1 TRP A  18      12.406  20.686  13.696  1.00 25.00           C  
ATOM    108  CD2 TRP A  18      13.872  21.190  15.308  1.00 25.00           C  
ATOM    109  NE1 TRP A  18      11.990  21.850  14.307  1.00 25.00           N  
ATOM    110  CE2 TRP A  18      12.881  22.180  15.288  1.00 25.00           C  
ATOM    111  CE3 TRP A  18      14.905  21.302  16.248  1.00 25.00           C  
ATOM    112  CZ2 TRP A  18      12.905  23.277  16.132  1.00 25.00           C  
ATOM    113  CZ3 TRP A  18      14.924  22.401  17.083  1.00 25.00           C  
ATOM    114  CH2 TRP A  18      13.902  23.345  17.048  1.00 25.00           C  
ATOM    115  N   PRO A  19      12.713  16.528  12.673  1.00 25.00           N  
ATOM    116  CA  PRO A  19      11.511  16.026  11.995  1.00 25.00           C  
ATOM    117  C   PRO A  19      10.872  14.864  12.696  1.00 25.00           C  
ATOM    118  O   PRO A  19       9.731  14.507  12.383  1.00 25.00           O  
ATOM    119  CB  PRO A  19      12.051  15.620  10.642  1.00 25.00           C  
ATOM    120  CG  PRO A  19      13.495  15.443  10.791  1.00 25.00           C  
ATOM    121  CD  PRO A  19      13.932  16.368  11.855  1.00 25.00           C  
ATOM    122  OXT PRO A  19      11.476  14.250  13.554  1.00 25.00           O
TER     123      PRO A  19
END
"""

def show(prefix, fmodel, m1, m2):
  s1 = m1.get_sites_cart()
  s2 = m2.get_sites_cart()
  d = flex.mean(flex.sqrt((s1 - s2).dot()))
  print "%s r_work=%6.4f r_free=%6.4f dist_to_answer=%6.4f"%(
    prefix, fmodel.r_work(), fmodel.r_free(), d)

def get_map(fmodel):
  map_coeffs = map_tools.electron_density_map(fmodel = fmodel).map_coefficients(
    map_type     = "2mFo-DFc",
    isotropize   = True,
    fill_missing = False)
  crystal_gridding = fmodel.f_obs().crystal_gridding(
    d_min             = fmodel.f_obs().d_min(),
    symmetry_flags    = maptbx.use_space_group_symmetry,
    resolution_factor = 1./4)
  fft_map = miller.fft_map(
    crystal_gridding     = crystal_gridding,
    fourier_coefficients = map_coeffs)
  fft_map.apply_sigma_scaling()
  return fft_map.real_map_unpadded()

def run():
  # Good answer model
  model_good = mmtbx.model.manager(
    model_input = iotbx.pdb.input(source_info=None, lines = pdb_str_answer),
    log         = null_out())
  of = open("model_good.pdb","w")
  of.write(model_good.model_as_pdb())
  of.close()
  # Poor model
  model_poor = mmtbx.model.manager(
    model_input = iotbx.pdb.input(source_info=None, lines = pdb_str_poor),
    build_grm   = True,
    log         = null_out())
  of = open("model_poor.pdb","w")
  of.write(model_poor.model_as_pdb())
  of.close()
  # Make up Fobs data and flags
  f_obs = abs(model_good.get_xray_structure().structure_factors(
    d_min=1.5).f_calc())
  r_free_flags = f_obs.generate_r_free_flags()
  # Set up fmodel
  fmodel = mmtbx.f_model.manager(
    xray_structure = model_poor.get_xray_structure(),
    f_obs          = f_obs,
    r_free_flags   = r_free_flags)
  show(prefix="Initial", fmodel=fmodel, m1=model_poor, m2=model_good)
  fmodel.update_all_scales()
  show(prefix="Bulk-solvent and scale", fmodel=fmodel, m1=model_poor, 
    m2=model_good)
  # Compute initial target map
  map_data = get_map(fmodel = fmodel)
  # Refinement loop
  states = mmtbx.utils.states(
    pdb_hierarchy  = model_poor.get_hierarchy(),
    xray_structure = model_poor.get_xray_structure())
  for macro_cycle in xrange(10):
    refined = mmtbx.refinement.real_space.individual_sites.easy(
      map_data                    = map_data,
      xray_structure              = model_poor.get_xray_structure(),
      pdb_hierarchy               = model_poor.get_hierarchy(),
      geometry_restraints_manager = model_poor.get_restraints_manager(),
      rms_bonds_limit             = 0.02,
      rms_angles_limit            = 2.0,
      max_iterations              = 50,
      states_accumulator          = states,
      selection                   = None,
      w                           = None,
      log                         = None)
    fmodel.update_xray_structure(
      xray_structure = refined.xray_structure,
      update_f_calc  = True,
      update_f_mask  = True)
    fmodel.update_all_scales()
    map_data = get_map(fmodel = fmodel)
    model_poor.set_sites_cart(sites_cart = refined.xray_structure.sites_cart())
    show(prefix="cycle: %3d"%macro_cycle, fmodel=fmodel, m1=model_poor, 
      m2=model_good)
    states.add(sites_cart = refined.xray_structure.sites_cart())
  # Result
  of = open("model_refined.pdb","w")
  of.write(model_poor.model_as_pdb())
  of.close()
  states.write(file_name="refined_all_states.pdb")

if (__name__ == "__main__"):
  run()
  print "OK"
