__all__ = ["kwargs_column_mag", "kwargs_column_coo", "kwargs_column_flags",
           "kwargs_column_el", "kwargs_column_fwhm", "kwargs_column_class",
           "prime_keywords_noboby_needs", "extension_keywords_noboby_needs"]

# =========================================================================== #
# Table format
# =========================================================================== #
kwargs_column_mag = dict(disp="F8.4", unit="mag")
kwargs_column_coo = dict(format="1D", disp="F11.7", unit="deg")
kwargs_column_flags = dict(format="1I", disp="I3")
kwargs_column_el = dict(format="1E", disp="F8.3")
kwargs_column_fwhm = dict(format="1E", disp="F7.4", unit="deg")
kwargs_column_class = dict(format="1E", disp="F6.3")


# =========================================================================== #
# Useless keywords that are removed during reading of headers
# =========================================================================== #
prime_keywords_noboby_needs = ["ORIGIN",
                               "LST",
                               "PI-COI",
                               "OBSERVER",
                               "COMMENT",
                               "ESO INS DATE",
                               "ESO INS FILT1 DATE",
                               "ESO INS FILT1 ENC",
                               "ESO INS FILT1 ERROR",
                               "ESO INS FILT1 FOCUS",
                               "ESO INS FILT1 ID",
                               "ESO INS FILT1 NO",
                               "ESO INS FILT1 POSEDGE",
                               "ESO INS FILT1 TRAYID",
                               "ESO INS FILT1 WLEN",
                               "ESO INS ID",
                               "ESO INS LSC1 OK",
                               "ESO INS LSC1 SWSIM",
                               "ESO INS LSM1 OK",
                               "ESO INS LSM1 SWSIM",
                               "ESO INS LSM2 OK",
                               "ESO INS LSM2 SWSIM",
                               "ESO INS LSM3 OK",
                               "ESO INS LSM3 SWSIM",
                               "ESO INS PRES1 ID",
                               "ESO INS PRES1 NAME",
                               "ESO INS PRES1 UNIT",
                               "ESO INS PRES1 VAL",
                               "ESO INS PRES2 ID",
                               "ESO INS PRES2 NAME",
                               "ESO INS PRES2 UNIT",
                               "ESO INS PRES2 VAL",
                               "ESO INS SW1 ID",
                               "ESO INS SW1 NAME",
                               "ESO INS SW1 STATUS",
                               "ESO INS SW2 ID",
                               "ESO INS SW2 NAME",
                               "ESO INS SW2 STATUS",
                               "ESO INS SW3 ID",
                               "ESO INS SW3 NAME",
                               "ESO INS SW3 STATUS",
                               "ESO INS TEMP1 ID",
                               "ESO INS TEMP1 NAME",
                               "ESO INS TEMP1 UNIT",
                               "ESO INS TEMP1 VAL",
                               "ESO INS TEMP10 ID",
                               "ESO INS TEMP10 NAME",
                               "ESO INS TEMP10 UNIT",
                               "ESO INS TEMP10 VAL",
                               "ESO INS TEMP12 ID",
                               "ESO INS TEMP12 NAME",
                               "ESO INS TEMP12 UNIT",
                               "ESO INS TEMP12 VAL",
                               "ESO INS TEMP14 ID",
                               "ESO INS TEMP14 NAME",
                               "ESO INS TEMP14 UNIT",
                               "ESO INS TEMP14 VAL",
                               "ESO INS TEMP15 ID",
                               "ESO INS TEMP15 NAME",
                               "ESO INS TEMP15 UNIT",
                               "ESO INS TEMP15 VAL",
                               "ESO INS TEMP16 ID",
                               "ESO INS TEMP16 NAME",
                               "ESO INS TEMP16 UNIT",
                               "ESO INS TEMP16 VAL",
                               "ESO INS TEMP17 ID",
                               "ESO INS TEMP17 NAME",
                               "ESO INS TEMP17 UNIT",
                               "ESO INS TEMP17 VAL",
                               "ESO INS TEMP18 ID",
                               "ESO INS TEMP18 NAME",
                               "ESO INS TEMP18 UNIT",
                               "ESO INS TEMP18 VAL",
                               "ESO INS TEMP19 ID",
                               "ESO INS TEMP19 NAME",
                               "ESO INS TEMP19 UNIT",
                               "ESO INS TEMP19 VAL",
                               "ESO INS TEMP2 ID",
                               "ESO INS TEMP2 NAME",
                               "ESO INS TEMP2 UNIT",
                               "ESO INS TEMP2 VAL",
                               "ESO INS TEMP20 ID",
                               "ESO INS TEMP20 NAME",
                               "ESO INS TEMP20 UNIT",
                               "ESO INS TEMP20 VAL",
                               "ESO INS TEMP21 ID",
                               "ESO INS TEMP21 NAME",
                               "ESO INS TEMP21 UNIT",
                               "ESO INS TEMP21 VAL",
                               "ESO INS TEMP22 ID",
                               "ESO INS TEMP22 NAME",
                               "ESO INS TEMP22 UNIT",
                               "ESO INS TEMP22 VAL",
                               "ESO INS TEMP23 ID",
                               "ESO INS TEMP23 NAME",
                               "ESO INS TEMP23 UNIT",
                               "ESO INS TEMP23 VAL",
                               "ESO INS TEMP24 ID",
                               "ESO INS TEMP24 NAME",
                               "ESO INS TEMP24 UNIT",
                               "ESO INS TEMP24 VAL",
                               "ESO INS TEMP25 ID",
                               "ESO INS TEMP25 NAME",
                               "ESO INS TEMP25 UNIT",
                               "ESO INS TEMP25 VAL",
                               "ESO INS TEMP26 ID",
                               "ESO INS TEMP26 NAME",
                               "ESO INS TEMP26 UNIT",
                               "ESO INS TEMP26 VAL",
                               "ESO INS TEMP3 ID",
                               "ESO INS TEMP3 NAME",
                               "ESO INS TEMP3 UNIT",
                               "ESO INS TEMP3 VAL",
                               "ESO INS TEMP4 ID",
                               "ESO INS TEMP4 NAME",
                               "ESO INS TEMP4 UNIT",
                               "ESO INS TEMP4 VAL",
                               "ESO INS TEMP5 ID",
                               "ESO INS TEMP5 NAME",
                               "ESO INS TEMP5 UNIT",
                               "ESO INS TEMP5 VAL",
                               "ESO INS TEMP6 ID",
                               "ESO INS TEMP6 NAME",
                               "ESO INS TEMP6 UNIT",
                               "ESO INS TEMP6 VAL",
                               "ESO INS TEMP7 ID",
                               "ESO INS TEMP7 NAME",
                               "ESO INS TEMP7 UNIT",
                               "ESO INS TEMP7 VAL",
                               "ESO INS TEMP8 ID",
                               "ESO INS TEMP8 NAME",
                               "ESO INS TEMP8 UNIT",
                               "ESO INS TEMP8 VAL",
                               "ESO INS THERMAL AMB MEAN",
                               "ESO INS THERMAL CLD MEAN",
                               "ESO INS THERMAL DET MEAN",
                               "ESO INS THERMAL DET TARGET",
                               "ESO INS THERMAL ENABLE",
                               "ESO INS THERMAL FPA MEAN",
                               "ESO INS THERMAL TUB MEAN",
                               "ESO INS THERMAL WIN MEAN",
                               "ESO INS VAC1 OK",
                               "ESO INS VAC1 SWSIM",
                               "ESO OBS AIRM",
                               "ESO OBS AMBI FWHM",
                               "ESO OBS AMBI TRANS",
                               "ESO OBS ATM",
                               "ESO OBS CONTAINER ID",
                               "ESO OBS CONTAINER TYPE",
                               "ESO OBS CONTRAST",
                               "ESO OBS DID",
                               "ESO OBS EXECTIME",
                               "ESO OBS GRP",
                               "ESO OBS MOON DIST",
                               "ESO OBS MOON FLI",
                               "ESO OBS NTPL",
                               "ESO OBS OBSERVER",
                               "ESO OBS PI-COI ID",
                               "ESO OBS PI-COI NAME",
                               "ESO OBS STREHLRATIO",
                               "ESO OBS TPLNO",
                               "ESO OBS TWILIGHT",
                               "ESO OBS WATERVAPOUR",
                               "ESO OCS DET1 IMGNAME",
                               "ESO OCS EXPNO",
                               "ESO OCS NEXP",
                               "ESO OCS RECIPE",
                               "ESO OCS REQTIME",
                               "ESO OCS SADT AOSA1 ID",
                               "ESO OCS SADT AOSA2 ID",
                               "ESO OCS SADT AOSA3 ID",
                               "ESO OCS SADT AOSA4 ID",
                               "ESO OCS SADT AOSA5 ID",
                               "ESO OCS SADT AOSB1 ID",
                               "ESO OCS SADT AOSB2 ID",
                               "ESO OCS SADT AOSB3 ID",
                               "ESO OCS SADT AOSB4 ID",
                               "ESO OCS SADT AOSB5 ID",
                               "ESO OCS SADT AREA ID",
                               "ESO OCS SADT CAT ID",
                               "ESO OCS SADT GS1 ID",
                               "ESO OCS SADT GS2 ID",
                               "ESO OCS SADT GS3 ID",
                               "ESO OCS SADT GS4 ID",
                               "ESO OCS SADT GS5 ID",
                               "ESO OCS SADT ID",
                               "ESO OCS SADT IP ID",
                               "ESO OCS SADT MAXJIT",
                               "ESO OCS SADT OVERLAPX",
                               "ESO OCS SADT OVERLAPY",
                               "ESO OCS SADT PATTERN",
                               "ESO OCS SADT TILE ID",
                               "ESO OCS TARG ALPHAOBJ",
                               "ESO OCS TARG DELTAOBJ",
                               "ESO OCS TARG X",
                               "ESO OCS TARG Y",
                               "ESO TEL ABSROT END",
                               "ESO TEL ABSROT START",
                               "ESO TEL AG REFX",
                               "ESO TEL AG REFY",
                               "ESO TEL AIRM END",
                               "ESO TEL AIRM START",
                               "ESO TEL ALT",
                               "ESO TEL AMBI FWHM END",
                               "ESO TEL AMBI FWHM START",
                               "ESO TEL AMBI PRES END",
                               "ESO TEL AMBI PRES START",
                               "ESO TEL AMBI RHUM",
                               "ESO TEL AMBI TAU0",
                               "ESO TEL AMBI TEMP",
                               "ESO TEL AMBI WINDDIR",
                               "ESO TEL AMBI WINDSP",
                               "ESO TEL AO ALT",
                               "ESO TEL AO DATE",
                               "ESO TEL AO M1 DATE",
                               "ESO TEL AO M2 DATE",
                               "ESO TEL AO MODES",
                               "ESO TEL AZ",
                               "ESO TEL DATE",
                               "ESO TEL DID",
                               "ESO TEL DID1",
                               "ESO TEL DOME STATUS",
                               "ESO TEL ECS FLATFIELD",
                               "ESO TEL ECS MOONSCR",
                               "ESO TEL ECS VENT1",
                               "ESO TEL ECS VENT2",
                               "ESO TEL ECS VENT3",
                               "ESO TEL ECS WINDSCR",
                               "ESO TEL FOCU ID",
                               "ESO TEL FOCU VALUE",
                               "ESO TEL GEOELEV",
                               "ESO TEL GEOLAT",
                               "ESO TEL GEOLON",
                               "ESO TEL GUID DEC",
                               "ESO TEL GUID FWHM",
                               "ESO TEL GUID ID",
                               "ESO TEL GUID MAG",
                               "ESO TEL GUID PEAKINT",
                               "ESO TEL GUID RA",
                               "ESO TEL GUID STATUS",
                               "ESO TEL ID",
                               "ESO TEL M2 ACENTRE",
                               "ESO TEL M2 ATILT",
                               "ESO TEL M2 BCENTRE",
                               "ESO TEL M2 BTILT",
                               "ESO TEL M2 Z",
                               "ESO TEL MOON DEC",
                               "ESO TEL MOON RA",
                               "ESO TEL OPER",
                               "ESO TEL PARANG END",
                               "ESO TEL PARANG START",
                               "ESO TEL POSANG",
                               "ESO TEL TARG ALPHA",
                               "ESO TEL TARG COORDTYPE",
                               "ESO TEL TARG DELTA",
                               "ESO TEL TARG EPOCH",
                               "ESO TEL TARG EPOCHSYSTEM",
                               "ESO TEL TARG EQUINOX",
                               "ESO TEL TARG PARALLAX",
                               "ESO TEL TARG PMA",
                               "ESO TEL TARG PMD",
                               "ESO TEL TARG RADVEL",
                               "ESO TEL TH M1 TEMP",
                               "ESO TEL TH STR TEMP",
                               "ESO TEL TRAK STATUS",
                               "ESO TPL DID",
                               "ESO TPL EXPNO",
                               "ESO TPL FILE DIRNAME",
                               "ESO TPL ID",
                               "ESO TPL NAME",
                               "ESO TPL NEXP",
                               "ESO TPL PRESEQ",
                               "ESO TPL START",
                               "ESO TPL VERSION",
                               "REQTIME", ]

extension_keywords_noboby_needs = ["ESO DET CHIP ID",
                                   "ESO DET CHIP LIVE",
                                   "ESO DET CHIP NAME",
                                   "ESO DET CHIP NO",
                                   "ESO DET CHIP NX",
                                   "ESO DET CHIP NY",
                                   "ESO DET CHIP PXSPACE",
                                   "ESO DET CHIP TYPE",
                                   "ESO DET CHIP VIGNETD",
                                   "ESO DET CHIP X",
                                   "ESO DET CHIP Y",
                                   "ESO DET CHOP FREQ",
                                   "ESO DET CON OPMODE",
                                   "ESO DET DID",
                                   "ESO DET DITDELAY",
                                   "ESO DET EXP NAME",
                                   "ESO DET EXP NO",
                                   "ESO DET EXP UTC",
                                   "ESO DET FILE CUBE ST",
                                   "ESO DET FRAM NO",
                                   "ESO DET FRAM TYPE",
                                   "ESO DET FRAM UTC",
                                   "ESO DET IRACE ADC1 DELAY",
                                   "ESO DET IRACE ADC1 ENABLE",
                                   "ESO DET IRACE ADC1 FILTER1",
                                   "ESO DET IRACE ADC1 FILTER2",
                                   "ESO DET IRACE ADC1 HEADER",
                                   "ESO DET IRACE ADC1 NAME",
                                   "ESO DET IRACE ADC10 DELAY",
                                   "ESO DET IRACE ADC10 ENABLE",
                                   "ESO DET IRACE ADC10 FILTER1",
                                   "ESO DET IRACE ADC10 FILTER2",
                                   "ESO DET IRACE ADC10 HEADER",
                                   "ESO DET IRACE ADC10 NAME",
                                   "ESO DET IRACE ADC11 DELAY",
                                   "ESO DET IRACE ADC11 ENABLE",
                                   "ESO DET IRACE ADC11 FILTER1",
                                   "ESO DET IRACE ADC11 FILTER2",
                                   "ESO DET IRACE ADC11 HEADER",
                                   "ESO DET IRACE ADC11 NAME",
                                   "ESO DET IRACE ADC12 DELAY",
                                   "ESO DET IRACE ADC12 ENABLE",
                                   "ESO DET IRACE ADC12 FILTER1",
                                   "ESO DET IRACE ADC12 FILTER2",
                                   "ESO DET IRACE ADC12 HEADER",
                                   "ESO DET IRACE ADC12 NAME",
                                   "ESO DET IRACE ADC13 DELAY",
                                   "ESO DET IRACE ADC13 ENABLE",
                                   "ESO DET IRACE ADC13 FILTER1",
                                   "ESO DET IRACE ADC13 FILTER2",
                                   "ESO DET IRACE ADC13 HEADER",
                                   "ESO DET IRACE ADC13 NAME",
                                   "ESO DET IRACE ADC14 DELAY",
                                   "ESO DET IRACE ADC14 ENABLE",
                                   "ESO DET IRACE ADC14 FILTER1",
                                   "ESO DET IRACE ADC14 FILTER2",
                                   "ESO DET IRACE ADC14 HEADER",
                                   "ESO DET IRACE ADC14 NAME",
                                   "ESO DET IRACE ADC15 DELAY",
                                   "ESO DET IRACE ADC15 ENABLE",
                                   "ESO DET IRACE ADC15 FILTER1",
                                   "ESO DET IRACE ADC15 FILTER2",
                                   "ESO DET IRACE ADC15 HEADER",
                                   "ESO DET IRACE ADC15 NAME",
                                   "ESO DET IRACE ADC16 DELAY",
                                   "ESO DET IRACE ADC16 ENABLE",
                                   "ESO DET IRACE ADC16 FILTER1",
                                   "ESO DET IRACE ADC16 FILTER2",
                                   "ESO DET IRACE ADC16 HEADER",
                                   "ESO DET IRACE ADC16 NAME",
                                   "ESO DET IRACE ADC2 DELAY",
                                   "ESO DET IRACE ADC2 ENABLE",
                                   "ESO DET IRACE ADC2 FILTER1",
                                   "ESO DET IRACE ADC2 FILTER2",
                                   "ESO DET IRACE ADC2 HEADER",
                                   "ESO DET IRACE ADC2 NAME",
                                   "ESO DET IRACE ADC3 DELAY",
                                   "ESO DET IRACE ADC3 ENABLE",
                                   "ESO DET IRACE ADC3 FILTER1",
                                   "ESO DET IRACE ADC3 FILTER2",
                                   "ESO DET IRACE ADC3 HEADER",
                                   "ESO DET IRACE ADC3 NAME",
                                   "ESO DET IRACE ADC4 DELAY",
                                   "ESO DET IRACE ADC4 ENABLE",
                                   "ESO DET IRACE ADC4 FILTER1",
                                   "ESO DET IRACE ADC4 FILTER2",
                                   "ESO DET IRACE ADC4 HEADER",
                                   "ESO DET IRACE ADC4 NAME",
                                   "ESO DET IRACE ADC5 DELAY",
                                   "ESO DET IRACE ADC5 ENABLE",
                                   "ESO DET IRACE ADC5 FILTER1",
                                   "ESO DET IRACE ADC5 FILTER2",
                                   "ESO DET IRACE ADC5 HEADER",
                                   "ESO DET IRACE ADC5 NAME",
                                   "ESO DET IRACE ADC6 DELAY",
                                   "ESO DET IRACE ADC6 ENABLE",
                                   "ESO DET IRACE ADC6 FILTER1",
                                   "ESO DET IRACE ADC6 FILTER2",
                                   "ESO DET IRACE ADC6 HEADER",
                                   "ESO DET IRACE ADC6 NAME",
                                   "ESO DET IRACE ADC7 DELAY",
                                   "ESO DET IRACE ADC7 ENABLE",
                                   "ESO DET IRACE ADC7 FILTER1",
                                   "ESO DET IRACE ADC7 FILTER2",
                                   "ESO DET IRACE ADC7 HEADER",
                                   "ESO DET IRACE ADC7 NAME",
                                   "ESO DET IRACE ADC8 DELAY",
                                   "ESO DET IRACE ADC8 ENABLE",
                                   "ESO DET IRACE ADC8 FILTER1",
                                   "ESO DET IRACE ADC8 FILTER2",
                                   "ESO DET IRACE ADC8 HEADER",
                                   "ESO DET IRACE ADC8 NAME",
                                   "ESO DET IRACE ADC9 DELAY",
                                   "ESO DET IRACE ADC9 ENABLE",
                                   "ESO DET IRACE ADC9 FILTER1",
                                   "ESO DET IRACE ADC9 FILTER2",
                                   "ESO DET IRACE ADC9 HEADER",
                                   "ESO DET IRACE ADC9 NAME",
                                   "ESO DET IRACE SEQCONT",
                                   "ESO DET MINDIT",
                                   "ESO DET MODE NAME",
                                   "ESO DET NCORRS",
                                   "ESO DET NCORRS NAME",
                                   "ESO DET NDITSKIP",
                                   "ESO DET RSPEED",
                                   "ESO DET RSPEEDADD",
                                   "ESO DET WIN NX",
                                   "ESO DET WIN NY",
                                   "ESO DET WIN STARTX",
                                   "ESO DET WIN STARTY",
                                   "ESO DET WIN TYPE",
                                   "PV2_1",
                                   "PV2_2",
                                   "PV2_3",
                                   "PV2_4",
                                   "PV2_5"]
