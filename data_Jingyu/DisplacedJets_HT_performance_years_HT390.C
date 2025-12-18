#ifdef __CLING__
#pragma cling optimize(0)
#endif
void DisplacedJets_HT_performance_years_HT390()
{
//=========Macro generated from canvas: can/can
//=========  (Wed Dec 17 20:22:27 2025) by ROOT version 6.34.04
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
   Double_t eff_graph_fy3001[64] = { 0, 4.20071e-05, 0.0002545541, 0.001015663, 0.002929261, 0.006931908, 0.01189061, 0.03272727, 0.05782012, 0.07548509, 0.08511763, 0.1537381, 0.2614596, 0.3143322, 0.3563344, 0.3893364, 0.5775606,
   0.8357466, 0.9704187, 0.984994, 0.9913255, 0.994186, 0.998234, 0.9983983, 0.9976704, 1, 1, 0.9982563, 0.9990148, 1, 1, 0.9986737, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9722222 };
   Double_t eff_graph_felx3001[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3001[64] = { 0, 2.010385e-05, 6.296265e-05, 0.0002308717, 0.0004252955, 0.000704273, 0.0009987198, 0.001766097, 0.003484172, 0.004078978, 0.004556625, 0.006087598, 0.007707482, 0.008478759, 0.009278308, 0.00969643, 0.01042678,
   0.008262884, 0.002932733, 0.002415137, 0.002093633, 0.001914553, 0.001394151, 0.001555567, 0.001838181, 0.00126799, 0.001419533, 0.002295143, 0.002261863, 0.001990464, 0.002220986, 0.003043102, 0.003003685,
   0.0032763, 0.003547788, 0.004204003, 0.004569185, 0.004709444, 0.005529899, 0.006484263, 0.007770591, 0.007250354, 0.008258599, 0.009744876, 0.01070846, 0.01203893, 0.01354463, 0.01315743, 0.008566006,
   0.0113, 0.01899459, 0.02835616, 0.02671064, 0.02750881, 0.01788732, 0.01690203, 0.020248, 0.02220132, 0.0368748, 0.04191086, 0.0368748, 0.02632867, 0.01616023, 0.06099571 };
   Double_t eff_graph_fehx3001[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3001[64] = { 3.448207e-06, 3.321327e-05, 8.085615e-05, 0.0002903537, 0.0004917893, 0.00077913, 0.001085159, 0.0018605, 0.003686313, 0.00428571, 0.004782324, 0.006286728, 0.00785417, 0.008602042, 0.009385281, 0.009783012, 0.0103595,
   0.0079504, 0.002685874, 0.002103294, 0.001718428, 0.001481877, 0.0008449969, 0.0008715823, 0.001114609, 0, 0, 0.001126176, 0.0008150354, 0, 0, 0.00109717, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02299056 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(64,eff_graph_fx3001,eff_graph_fy3001,eff_graph_felx3001,eff_graph_fehx3001,eff_graph_fely3001,eff_graph_fehy3001);
   grae->SetName("eff_graph");
   grae->SetTitle("");
   grae->SetFillColor(19);
   grae->SetMarkerStyle(20);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3001 = new TH1F("Graph_eff_graph3001","",100,0,3300);
   Graph_eff_graph3001->SetMinimum(0);
   Graph_eff_graph3001->SetMaximum(1.05);
   Graph_eff_graph3001->SetDirectory(nullptr);
   Graph_eff_graph3001->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
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
   Double_t eff_graph_fy3002[64] = { 0, 4.20071e-05, 0.0002545541, 0.001015663, 0.002929261, 0.006931908, 0.01189061, 0.03272727, 0.05782012, 0.07548509, 0.08511763, 0.1537381, 0.2614596, 0.3143322, 0.3563344, 0.3893364, 0.5775606,
   0.8357466, 0.9704187, 0.984994, 0.9913255, 0.994186, 0.998234, 0.9983983, 0.9976704, 1, 1, 0.9982563, 0.9990148, 1, 1, 0.9986737, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9722222 };
   Double_t eff_graph_felx3002[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3002[64] = { 0, 2.010385e-05, 6.296265e-05, 0.0002308717, 0.0004252955, 0.000704273, 0.0009987198, 0.001766097, 0.003484172, 0.004078978, 0.004556625, 0.006087598, 0.007707482, 0.008478759, 0.009278308, 0.00969643, 0.01042678,
   0.008262884, 0.002932733, 0.002415137, 0.002093633, 0.001914553, 0.001394151, 0.001555567, 0.001838181, 0.00126799, 0.001419533, 0.002295143, 0.002261863, 0.001990464, 0.002220986, 0.003043102, 0.003003685,
   0.0032763, 0.003547788, 0.004204003, 0.004569185, 0.004709444, 0.005529899, 0.006484263, 0.007770591, 0.007250354, 0.008258599, 0.009744876, 0.01070846, 0.01203893, 0.01354463, 0.01315743, 0.008566006,
   0.0113, 0.01899459, 0.02835616, 0.02671064, 0.02750881, 0.01788732, 0.01690203, 0.020248, 0.02220132, 0.0368748, 0.04191086, 0.0368748, 0.02632867, 0.01616023, 0.06099571 };
   Double_t eff_graph_fehx3002[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3002[64] = { 3.448207e-06, 3.321327e-05, 8.085615e-05, 0.0002903537, 0.0004917893, 0.00077913, 0.001085159, 0.0018605, 0.003686313, 0.00428571, 0.004782324, 0.006286728, 0.00785417, 0.008602042, 0.009385281, 0.009783012, 0.0103595,
   0.0079504, 0.002685874, 0.002103294, 0.001718428, 0.001481877, 0.0008449969, 0.0008715823, 0.001114609, 0, 0, 0.001126176, 0.0008150354, 0, 0, 0.00109717, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02299056 };
   grae = new TGraphAsymmErrors(64,eff_graph_fx3002,eff_graph_fy3002,eff_graph_felx3002,eff_graph_fehx3002,eff_graph_fely3002,eff_graph_fehy3002);
   grae->SetName("eff_graph");
   grae->SetTitle("");
   grae->SetFillColor(19);
   grae->SetMarkerStyle(20);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3002 = new TH1F("Graph_eff_graph3002","",100,0,3300);
   Graph_eff_graph3002->SetMinimum(0);
   Graph_eff_graph3002->SetMaximum(1.05);
   Graph_eff_graph3002->SetDirectory(nullptr);
   Graph_eff_graph3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3002->SetLineColor(ci);
   Graph_eff_graph3002->GetXaxis()->SetTitle("Offline calorimeter H_{T} [GeV]");
   Graph_eff_graph3002->GetXaxis()->SetRange(1,49);
   Graph_eff_graph3002->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetXaxis()->SetTitleSize(0.05);
   Graph_eff_graph3002->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3002->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3002->GetYaxis()->SetTitle("HLT efficiency");
   Graph_eff_graph3002->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3002->GetYaxis()->SetTitleSize(0.05);
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
   Double_t eff_graph_fy3003[64] = { 2.621639e-06, 9.039041e-05, 0.0007029146, 0.002356877, 0.006303467, 0.01478212, 0.0299493, 0.0704776, 0.1009052, 0.133088, 0.1943645, 0.286478, 0.3590773, 0.4079254, 0.492601, 0.6564516, 0.8364523,
   0.9414574, 0.9852183, 0.9907658, 0.9940844, 0.991898, 0.9945726, 0.9930762, 0.9956492, 0.9927086, 0.9953461, 0.9957181, 0.9942551, 0.9951089, 0.9969773, 0.9956044, 0.9938875,
   0.9941945, 0.994527, 0.9937276, 0.9948823, 0.9929988, 0.9975155, 0.9917695, 0.9938272, 0.9943396, 0.9960239, 0.9979123, 0.9977477, 0.9892761, 0.9919786, 0.9848024, 0.9921875,
   0.9977427, 1, 0.9772727, 0.9936709, 1, 0.9964539, 1, 0.9952381, 0.9939394, 1, 0.9923664, 0.9895833, 0.9923664, 0.9921569, 1 };
   Double_t eff_graph_felx3003[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fely3003[64] = { 1.254676e-06, 1.795584e-05, 6.276811e-05, 0.0002111616, 0.0003754285, 0.0006224488, 0.0009542721, 0.001556793, 0.002752807, 0.003229227, 0.00396035, 0.004658437, 0.005179854, 0.005493022, 0.005837752, 0.00584661, 0.004770443,
   0.003291259, 0.001288357, 0.001130608, 0.001003646, 0.001253489, 0.001135293, 0.001373577, 0.001241122, 0.001682987, 0.001473959, 0.001542779, 0.001891909, 0.001957038, 0.001801003, 0.002160545, 0.002596783,
   0.002850536, 0.002935421, 0.003362083, 0.003447251, 0.00415823, 0.003267411, 0.004883518, 0.004853828, 0.005475293, 0.005220101, 0.004784168, 0.005159895, 0.008397875, 0.007740966, 0.01015015, 0.006134065,
   0.0051715, 0.01000979, 0.01760506, 0.01440256, 0.01244586, 0.008106744, 0.007055842, 0.01086434, 0.01379768, 0.0118073, 0.01733356, 0.02354427, 0.01733356, 0.01025066, 0.02332647 };
   Double_t eff_graph_fehx3003[64] = { 50, 25, 25, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5,
   5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20,
   20, 10, 10, 10, 10, 20, 25, 25, 25, 25, 25, 25, 50, 200, 500 };
   Double_t eff_graph_fehy3003[64] = { 2.072895e-06, 2.193323e-05, 6.863522e-05, 0.0002309309, 0.000398324, 0.0006488006, 0.0009841752, 0.001588849, 0.002820097, 0.003295997, 0.004021758, 0.004703704, 0.005212557, 0.005515867, 0.005839748, 0.005800149, 0.00466284,
   0.003130691, 0.001190127, 0.001013938, 0.0008671587, 0.001095789, 0.0009526405, 0.001161933, 0.0009880079, 0.00139116, 0.001149684, 0.001170842, 0.001464311, 0.001450279, 0.001198323, 0.001519692, 0.001897459,
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
   grae->SetMarkerStyle(23);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3003 = new TH1F("Graph_eff_graph3003","eff_HT_disp_430_variable",100,0,3300);
   Graph_eff_graph3003->SetMinimum(1.230267e-06);
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
   
   TLegend *leg = new TLegend(0.42,0.25,0.86,0.5,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(0);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("eff_graph","#splitline{Data 2023 before HCAL}{conditions update}","ple");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph","#splitline{Data 2023 after HCAL}{conditions update}","ple");

   ci = TColor::GetColor("#0000ff");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(23);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   leg->Draw();
   TLatex *   tex = new TLatex(0.18,0.82,"CMS");
   tex->SetNDC();
   tex->SetTextSize(0.06);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.42,0.92,"27.9 fb^{-1} (2023) (13.6 TeV)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.045);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.45,0.6,"HLT calorimeter H_{T} > 390 GeV");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.035);
   tex->SetLineWidth(2);
   tex->Draw();
   can->Modified();
   can->SetSelected(can);
}
