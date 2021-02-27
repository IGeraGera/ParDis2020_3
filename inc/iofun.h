
void checkPatchSize(int patchSize);
void checkArgsNum(int num,int argc);
int getAttributes(FILE *fp);
double ** readCSV(FILE *fp, double **data, int *dataRows);
float *getImg( char *filename, int *dataRows, int *attributes );
void writeImg(char *in_filename,float *data, int dataRows,int attributes);

