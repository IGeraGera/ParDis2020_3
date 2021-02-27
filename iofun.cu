#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
extern int errno;
/* Function that saves the Image  */
void
writeImg(char *in_filename,float *data, int dataRows,int attributes){
	FILE *fp;
	int counter=1;
	char delim = ',';
	char *filename;
	filename = strtok(in_filename,".");
	filename = strcat(filename,"_out.csv"); 
	/* Check if the out file existsi and find an appropriate name that is unique*/
	while (access(filename,F_OK) == 0){
		filename = strtok(filename,"_");
		filename = strcat(filename,"_out");
		char num[100];
		sprintf(num,"%d",counter);
		counter++;
		filename = strcat(filename, num);
		filename = strcat(filename,".csv");
	}
	/* Open File */	
	fp = fopen(filename,"w");
	/* Write contents */
	for (int i=0;i<dataRows;i++){
		for (int j=0;j<attributes;j++){
			/* If writing the last element of the row skip the delimiter and add new line instead*/
			if ( j == attributes-1)
				fprintf(fp,"%f\n",data[i * attributes +j]);
			else
				fprintf(fp,"%f%c",data[i * attributes +j],delim);
		}

		
	}
	/* Close File*/
	fclose(fp);
}

/* Function that checks the number of arguments */
void
checkArgsNum(int num,int argc){
	num++; // Add 1 to offset
	if (num ==  argc ) return;
	else if (argc == 1 ){
		fprintf(stderr, "Line:%d No argument was given\n",__LINE__);
		exit(EXIT_FAILURE);
	}
	else if (argc < num && argc > 1){
		fprintf(stderr, "Line:%d Less arguments was given\n",__LINE__);
		exit(EXIT_FAILURE);
	}
	else if (argc > num){
		fprintf(stderr, "Line:%d More args given\n",__LINE__);
		exit(EXIT_FAILURE);
	}
}
/* Check if patch size given as an argument is an odd number */
void checkPatchSize(int patchSize){
	if ( (patchSize % 2)!=1){
		fprintf(stderr, "Line:%d Patch Size is not odd number\n",__LINE__);
		exit(EXIT_FAILURE);
	}
}
/* Helper function that gets the number of attributes in the csv file */
int
getAttributes(FILE *fp){
	/* Variables */
	char buf[2048];
	int attributes=0;
	char delim=',';
	/* Get first line from file */
	/* and check if file is empty */
	if (fgets(buf,2048,fp)==NULL){
		fprintf(stderr,"Line %d: Input file empty\n",__LINE__ );
		exit(EXIT_FAILURE);
	}
	/* Count the occurences of the delimiters */
	for(int i=0;i<strlen(buf);i++){
		if (buf[i]==delim) attributes++;
	}
	/* Add 1 to the occurrences of the delimiter */
	attributes++;
	/* Rewind fp */
	rewind(fp);
	return attributes;
}
/* Function that reads csv files and returns a 2d array  */
float **
readCSV(FILE *fp, float **data, int *dataRows,int *attributes){
	/* Variables */
	*attributes = getAttributes(fp);
	const char delim[] = ",";
	int row=0;
	char buf[2048];
	/* Read Lines one by one and split them on the delimiter */
	while(fgets(buf,2048,fp)){
		/* realloc the data array to fill another row */
		data = (float **)realloc(data,(row+1)*sizeof(float *));
		data[row] = (float *)malloc(*attributes*(sizeof(float)));
		/* Split the buf on the delimiter and fill the row */
		char *token;
		for (int i = 0; i< *attributes; i++){
			if (i==0)
				token = strtok(buf,delim); 
			else
				token = strtok(NULL,delim);
			/* If token NULL no more lines exist (Maybe there is no need for this) */
			if (token==NULL) break;
			/* Covert str to float */
			sscanf(token, "%f",&data[row][i]);
		}
		row++;
	}
	/* Return dataRows */
	*dataRows=row;
	rewind(fp);
	return data;
}
/* Wrapper function that calls the pre-existing readCSV and returns the data as a 1D array */
float *
getImg( char *filename, int *dataRows, int *attributes ){

	FILE *fp;
	/* Open file */
	fp= fopen (filename,"r");
	if (fp == NULL){
		fprintf(stderr, "Line %d: Error opening file %s\n",__LINE__,strerror(errno));
		exit (EXIT_FAILURE);
	}
	/* Read csv */
	float **data2D= (float **)malloc(0);
	data2D=readCSV(fp,data2D,dataRows,attributes);
	/* Make 2D array to 1D */
	float *data1D = (float *)malloc(*attributes * (*dataRows) * sizeof(float));
	for (int i=0;i<(*dataRows);i++)
		for (int j=0;j<(*attributes);j++)
			data1D[i*(*attributes) + j] = data2D[i][j];

	/* Free  data memory*/
	int RowsToFree=*dataRows;
	while(RowsToFree) free(data2D[-- RowsToFree]);
	free(data2D);
	/* Close file */
	fclose(fp);
	return data1D;
}

