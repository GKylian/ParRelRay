#pragma once
#include <iostream>
#include <fstream>
#include "utils.h"


using namespace std;



void readParams(string fname, map2d *params) {
	ifstream input(fname);
	cout << "Reading ray tracing parameters from .params file..." << endl;
	int count = 0;

	string block = ""; /* What block are we in (if any) ? */
	for (string line; getline(input, line);) {
		if (!(line.find('_') != string::npos) && !(line.find('<') != string::npos)) continue; /* Line without any parameter or block name -> skip */
		if (line.find('_') != string::npos) {
			unsigned first = line.find('_');
			unsigned last = line.find_last_of('_');
			block = line.substr(first+1, last-first-1);
			cout << "\tEntering block named " << block << endl;
			continue;
		}
		/* Split the line in two (before and after the equal */
		string left = line.substr(0, line.find('='));
		string right = line.substr(line.find('=')+1, line.length()-1);

		string key = ""; string val = "";
		if (left != "") {
			unsigned first = left.find('<'); unsigned last = left.find('>');
			key = left.substr(first+1, last-first-1);
		}
		if (right != "") {
			unsigned first = right.find('<'); unsigned last = right.find('>');
			val = right.substr(first+1, last-first-1);
			count++;
		}
		


		(*params)[block][key] = val;
	}

	std::cout << "Read " << count << " parameters" << endl;
	
}