* getopt
```cpp
 85     gboolean useDisplay = FALSE;
 86     guint tiler_rows, tiler_cols;
 87     guint batchSize = 1;
 88     guint pgie_batch_size;
 89     guint c;
 90     const char* optStr = "b:c:dhi:";
 91     std::string pgie_config;
 92     std::string input_file;
 93 
 94     while ((c = getopt(argc, argv, optStr)) != -1) {
 95         switch (c) {
 96             case 'b':
 97                 batchSize = std::atoi(optarg);
 98                 batchSize = batchSize == 0 ? 1:batchSize;
 99                 break;
100             case 'c':
101                 pgie_config.assign(optarg);
102                 break;
103             case 'd':
104                 useDisplay = TRUE;
105                 break;
106             case 'i':
107                 input_file.assign(optarg);
108                 break;
109             case 'h':
110             default:
111                 printUsage(argv[0]);
112                 return -1;
113           }
114      }


```
refer [offical link](https://linux.die.net/man/3/getopt_long) [link1](https://www.jianshu.com/p/80cdbf718916)
