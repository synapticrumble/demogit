import clr
import gc
from easyxlsx import *

clr.AddReference('easyxlsx')


# file path: C:\\demogit\\KeysExport.xml

workbook = ExcelDocument()
print("Reading file C:\\demogit\\KeysExport.xml")

if workbook.easy_LoadXMLSpreadsheetFile("C:\\demogit\\KeysExport.xml"):    
    xlsSecondTable = workbook.easy_getSheet("Second tab").easy_getExcelTable()    

    for column in range(5):
        xlsSecondTable.easy_getCell(1, column).setValue("Data " + str(column + 1))

    workbook.easy_WriteXLSXFile("C:\\demogit\\result.xlsx")

    sError = workbook.easy_getError()

    if sError == "":
        print("\nFile successfully created.\n\n")
    else:
            print("\nError encountered: " + sError + "\n\n")
else:
    print("\nError reading file D:/1.xml" + workbook.easy_getError() + "\n\n")


gc.collect()