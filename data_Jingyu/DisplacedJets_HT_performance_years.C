#ifdef __CLING__
#pragma cling optimize(0)
#endif
void DisplacedJets_HT_performance_years()
{
//=========Macro generated from canvas: can/can
//=========  (Wed Dec 17 20:22:13 2025) by ROOT version 6.34.04
   TCanvas *can = new TCanvas("can", "can",0,62,800,800);
   can->SetHighLightColor(2);
   can->Range(-297.8684,-0.21,1829.763,1.19);
   can->SetFillColor(0);
   can->SetBorderMode(0);
   can->SetBorderSize(2);
   can->SetTickx(1);
   can->SetTicky(1);
   can->SetLeftMargin(0.14);
   can->SetBottomMargin(0.15);
   can->SetFrameBorderMode(0);
   can->SetFrameBorderMode(0);
   
   Double_t eff_graph_fx3001[64] = { 50, 125, 175, 210, 230, 250, 270, 290, 305, 315, 325, 335, 345, 355, 365, 375, 385,
   395, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690,
   710, 730, 750, 770, 790, 810, 830, 850, 870, 890, 910, 930, 950, 970, 990, 1020,
   1060, 1090, 1110, 1130, 1150, 1180, 1225, 1275, 1325, 1375, 1425, 1475, 1550, 1800, 2500 };
   Double_t eff_graph_fy3001[64] = { 3.958679e-09, 1.380218e-06, 2.230167e-05, 0.000105772, 0.0003698548, 0.0008674409, 0.002430514, 0.006937325, 0.01014858, 0.01567263, 0.0290488, 0.04912937, 0.06223141, 0.0738323, 0.1003307, 0.1763921, 0.2788297,
   0.331699, 0.391218, 0.7613012, 0.9823282, 0.9940286, 0.9963665, 0.9972865, 0.998074, 0.9985881, 0.9986865, 0.9989246, 0.9990863, 0.9994282, 0.9995278, 0.9994726, 0.9997992,
   0.9998224, 0.9997489, 1, 0.9998721, 1, 0.9997617, 1, 1, 0.99978, 1, 1, 1, 1, 1, 1, 1,
   1, 0.9996579, 1, 1, 1, 1, 1, 0.9997026, 1, 1, 1, 1, 1, 1, 0.9974619 };
   Double_t eff_graph_felx3001[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3001[64] = { 2.55694e-09, 3.523585e-07, 2.041496e-06, 8.896644e-06, 1.874121e-05, 3.221779e-05, 5.999664e-05, 0.0001125107, 0.0002071262, 0.0002690323, 0.0003832905, 0.0005155195, 0.0006050158, 0.0006866996, 0.0008253976, 0.001091983, 0.001342592,
   0.001474659, 0.001151339, 0.001093722, 0.0003726446, 0.0002393495, 0.0002040834, 0.000193406, 0.0001781174, 0.0001680426, 0.0001736264, 0.0001725427, 0.0001734644, 0.0001539188, 0.0001560383, 0.0001742967, 0.0001357867,
   0.0001404383, 0.0001698012, 0.0001028623, 0.0001686361, 0.000130718, 0.0002317499, 0.0001619631, 0.0001796133, 0.0002901515, 0.0002238074, 0.0002453413, 0.0002738833, 0.0002995518, 0.0003444432, 0.0003651429, 0.0002152255,
   0.0002611037, 0.0007862569, 0.00064216, 0.0007155377, 0.0007648867, 0.00042197, 0.0004297516, 0.0006836402, 0.0006530945, 0.0007928874, 0.0009529489, 0.001142129, 0.0007677575, 0.0004199487, 0.002002263 };
   Double_t eff_graph_fehx3001[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3001[64] = { 5.221221e-09, 4.562756e-07, 2.237504e-06, 9.678283e-06, 1.971541e-05, 3.343609e-05, 6.149245e-05, 0.0001143373, 0.0002113522, 0.0002736145, 0.0003882264, 0.0005206723, 0.000610529, 0.0006925982, 0.0008314474, 0.001097301, 0.001346555,
   0.001477957, 0.001152548, 0.001090302, 0.0003651613, 0.0002303455, 0.0001935318, 0.0001809543, 0.0001636214, 0.0001509904, 0.0001543564, 0.0001500253, 0.0001475126, 0.0001237479, 0.0001205226, 0.0001346281, 8.671103e-05,
   8.501563e-05, 0.0001084359, 0, 8.259642e-05, 0, 0.0001296902, 0, 0, 0.0001421282, 0, 0, 0, 0, 0, 0, 0,
   0, 0.0002830145, 0, 0, 0, 0, 0, 0.0002460591, 0, 0, 0, 0, 0, 0, 0.001214298 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(64,eff_graph_fx3001,eff_graph_fy3001,eff_graph_felx3001,eff_graph_fehx3001,eff_graph_fely3001,eff_graph_fehy3001);
   grae->SetName("eff_graph");
   grae->SetTitle("");
   grae->SetFillColor(19);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#00cc00");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#00cc00");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(20);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3001 = new TH1F("Graph_eff_graph3001","",100,0,3300);
   Graph_eff_graph3001->SetMinimum(0);
   Graph_eff_graph3001->SetMaximum(1.05);
   Graph_eff_graph3001->SetDirectory(nullptr);
   Graph_eff_graph3001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3001->SetLineColor(ci);
   Graph_eff_graph3001->GetXaxis()->SetTitle("Offline calorimeter H_{T} [GeV]");
   Graph_eff_graph3001->GetXaxis()->SetRange(1,49);
   Graph_eff_graph3001->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetXaxis()->SetTitleSize(0.05);
   Graph_eff_graph3001->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3001->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetYaxis()->SetTitle("HLT efficiency");
   Graph_eff_graph3001->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetYaxis()->SetTitleSize(0.05);
   Graph_eff_graph3001->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3001);
   
   grae->Draw("ape");
   
   Double_t eff_graph_fx3002[64] = { 50, 125, 175, 210, 230, 250, 270, 290, 305, 315, 325, 335, 345, 355, 365, 375, 385,
   395, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690,
   710, 730, 750, 770, 790, 810, 830, 850, 870, 890, 910, 930, 950, 970, 990, 1020,
   1060, 1090, 1110, 1130, 1150, 1180, 1225, 1275, 1325, 1375, 1425, 1475, 1550, 1800, 2500 };
   Double_t eff_graph_fy3002[64] = { 0, 0, 3.181927e-05, 0.0001603678, 0.000436273, 0.00166077, 0.002972652, 0.007764128, 0.01063358, 0.01822054, 0.02669839, 0.05424396, 0.0805187, 0.07361564, 0.09127419, 0.1668585, 0.279218,
   0.3457014, 0.3920782, 0.7079832, 0.9753643, 0.9930233, 0.9969095, 0.9978644, 0.9976704, 1, 1, 0.9982563, 0.9980296, 1, 1, 0.9986737, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9722222 };
   Double_t eff_graph_felx3002[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3002[64] = { 0, 0, 2.055227e-05, 8.72765e-05, 0.0001608758, 0.00034358, 0.0004995436, 0.0008690083, 0.001523486, 0.002057004, 0.002623246, 0.003808616, 0.004745751, 0.004732895, 0.005531613, 0.007362302, 0.009363111,
   0.01026944, 0.007826863, 0.008089632, 0.003238367, 0.002053077, 0.001660625, 0.001685375, 0.001838181, 0.00126799, 0.001419533, 0.002295143, 0.002592942, 0.001990464, 0.002220986, 0.003043102, 0.003003685,
   0.0032763, 0.003547788, 0.004204003, 0.004569185, 0.004709444, 0.005529899, 0.006484263, 0.007770591, 0.007250354, 0.008258599, 0.009744876, 0.01070846, 0.01203893, 0.01354463, 0.01315743, 0.008566006,
   0.0113, 0.01899459, 0.02835616, 0.02671064, 0.02750881, 0.01788732, 0.01690203, 0.020248, 0.02220132, 0.0368748, 0.04191086, 0.0368748, 0.02632867, 0.01616023, 0.06099571 };
   Double_t eff_graph_fehx3002[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3002[64] = { 3.448207e-06, 1.933381e-05, 4.196583e-05, 0.0001559696, 0.0002349028, 0.0004230718, 0.000591131, 0.0009712321, 0.001756044, 0.002298115, 0.00288607, 0.004068422, 0.005006789, 0.005020223, 0.005839903, 0.007624532, 0.009555034,
   0.01041216, 0.007881842, 0.007962437, 0.002888696, 0.001625699, 0.00113904, 0.001021799, 0.001114609, 0, 0, 0.001126176, 0.001272623, 0, 0, 0.00109717, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02299056 };
   grae = new TGraphAsymmErrors(64,eff_graph_fx3002,eff_graph_fy3002,eff_graph_felx3002,eff_graph_fehx3002,eff_graph_fely3002,eff_graph_fehy3002);
   grae->SetName("eff_graph");
   grae->SetTitle("eff_HT_disp_430_variable");
   grae->SetFillColor(19);
   grae->SetMarkerStyle(21);
   
   TH1F *Graph_eff_graph3002 = new TH1F("Graph_eff_graph3002","eff_HT_disp_430_variable",100,0,3300);
   Graph_eff_graph3002->SetMinimum(0);
   Graph_eff_graph3002->SetMaximum(1.1);
   Graph_eff_graph3002->SetDirectory(nullptr);
   Graph_eff_graph3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3002->SetLineColor(ci);
   Graph_eff_graph3002->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3002->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3002->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3002->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3002);
   
   grae->Draw("pe ");
   
   Double_t eff_graph_fx3003[64] = { 50, 125, 175, 210, 230, 250, 270, 290, 305, 315, 325, 335, 345, 355, 365, 375, 385,
   395, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690,
   710, 730, 750, 770, 790, 810, 830, 850, 870, 890, 910, 930, 950, 970, 990, 1020,
   1060, 1090, 1110, 1130, 1150, 1180, 1225, 1275, 1325, 1375, 1425, 1475, 1550, 1800, 2500 };
   Double_t eff_graph_fy3003[64] = { 1.31082e-06, 1.446247e-05, 0.0001574529, 0.0006272333, 0.001215669, 0.003030867, 0.007291732, 0.01611654, 0.02724026, 0.0420892, 0.05903363, 0.08238994, 0.1079871, 0.1430499, 0.2174377, 0.3079179, 0.3796194,
   0.4189954, 0.598269, 0.8953829, 0.9870113, 0.9899475, 0.9940638, 0.9928783, 0.9956492, 0.9924386, 0.9950553, 0.9957181, 0.9942551, 0.9951089, 0.9964736, 0.9956044, 0.9938875,
   0.9941945, 0.994527, 0.9937276, 0.9948823, 0.9929988, 0.9975155, 0.9917695, 0.9938272, 0.9943396, 0.9960239, 0.9979123, 0.9977477, 0.9892761, 0.9919786, 0.9848024, 0.9921875,
   0.9977427, 1, 0.9772727, 0.9936709, 1, 0.9964539, 1, 0.9952381, 0.9939394, 1, 0.9923664, 0.9895833, 0.9923664, 0.9921569, 1 };
   Double_t eff_graph_felx3003[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3003[64] = { 8.466679e-07, 6.921504e-06, 2.957549e-05, 0.0001086079, 0.0001648428, 0.0002830967, 0.0004758316, 0.0007650088, 0.001484909, 0.001905174, 0.002352706, 0.002822985, 0.003336403, 0.003894854, 0.004791091, 0.00563431, 0.006144661,
   0.006635031, 0.004892592, 0.00334949, 0.001414993, 0.001376331, 0.001178353, 0.001389805, 0.001241122, 0.001708057, 0.001508699, 0.001542779, 0.001891909, 0.001957038, 0.001894232, 0.002160545, 0.002596783,
   0.002850536, 0.002935421, 0.003362083, 0.003447251, 0.00415823, 0.003267411, 0.004883518, 0.004853828, 0.005475293, 0.005220101, 0.004784168, 0.005159895, 0.008397875, 0.007740966, 0.01015015, 0.006134065,
   0.0051715, 0.01000979, 0.01760506, 0.01440256, 0.01244586, 0.008106744, 0.007055842, 0.01086434, 0.01379768, 0.0118073, 0.01733356, 0.02354427, 0.01733356, 0.01025066, 0.02332647 };
   Double_t eff_graph_fehx3003[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3003[64] = { 1.728876e-06, 1.143515e-05, 3.572961e-05, 0.000129248, 0.0001888328, 0.0003107635, 0.000507685, 0.0008015608, 0.001565631, 0.001989299, 0.002442053, 0.00291216, 0.003427924, 0.003983818, 0.004867479, 0.005691372, 0.006183008,
   0.006664068, 0.004873288, 0.003258901, 0.001283458, 0.001220278, 0.0009965375, 0.001178476, 0.0009880079, 0.001416851, 0.001185685, 0.001170842, 0.001464311, 0.001450279, 0.001299601, 0.001519692, 0.001897459,
   0.002006547, 0.00201621, 0.002310334, 0.002208805, 0.002773639, 0.001604579, 0.003259937, 0.002951951, 0.003079031, 0.002567811, 0.001727093, 0.001863243, 0.005125394, 0.004362418, 0.006549104, 0.003735295,
   0.001867449, 0, 0.01084586, 0.005236333, 0, 0.002933685, 0, 0.003939606, 0.005014161, 0, 0.006315726, 0.008618766, 0.006315726, 0.005064345, 0 };
   grae = new TGraphAsymmErrors(64,eff_graph_fx3003,eff_graph_fy3003,eff_graph_felx3003,eff_graph_fehx3003,eff_graph_fely3003,eff_graph_fehy3003);
   grae->SetName("eff_graph");
   grae->SetTitle("eff_HT_disp_430_variable");
   grae->SetFillColor(19);

   ci = TColor::GetColor("#0000ff");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#0000ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(22);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3003 = new TH1F("Graph_eff_graph3003","eff_HT_disp_430_variable",100,0,3300);
   Graph_eff_graph3003->SetMinimum(4.177365e-07);
   Graph_eff_graph3003->SetMaximum(1.1);
   Graph_eff_graph3003->SetDirectory(nullptr);
   Graph_eff_graph3003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3003->SetLineColor(ci);
   Graph_eff_graph3003->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3003->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3003->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3003->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3003->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3003);
   
   grae->Draw("pe ");
   
   TLegend *leg = new TLegend(0.42,0.2,0.86,0.55,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(0);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("eff_graph","Data 2022","ple");

   ci = TColor::GetColor("#00cc00");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#00cc00");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph","#splitline{Data 2023 before HCAL}{conditions update}","ple");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph","#splitline{Data 2023 after HCAL}{conditions update}","ple");

   ci = TColor::GetColor("#0000ff");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(22);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   leg->Draw();
   TLatex *   tex = new TLatex(0.18,0.82,"CMS");
   tex->SetNDC();
   tex->SetTextSize(0.06);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.16,0.92,"34.7 fb^{-1} (2022) + 27.9 fb^{-1} (2023) (13.6 TeV)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.042);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.45,0.6,"HLT calorimeter H_{T} > 430 GeV");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.035);
   tex->SetLineWidth(2);
   tex->Draw();
   can->Modified();
   can->SetSelected(can);
}
