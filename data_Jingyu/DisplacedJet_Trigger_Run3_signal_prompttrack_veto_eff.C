#ifdef __CLING__
#pragma cling optimize(0)
#endif
void DisplacedJet_Trigger_Run3_signal_prompttrack_veto_eff()
{
//=========Macro generated from canvas: can/can
//=========  (Wed Dec 17 20:26:07 2025) by ROOT version 6.34.04
   TCanvas *can = new TCanvas("can", "can",0,62,800,800);
   gStyle->SetOptStat(0);
   can->SetHighLightColor(2);
   can->Range(-2.906154,-0.1615385,16.47846,1.184615);
   can->SetFillColor(0);
   can->SetBorderMode(0);
   can->SetBorderSize(2);
   can->SetTickx(1);
   can->SetTicky(1);
   can->SetLeftMargin(0.12);
   can->SetBottomMargin(0.12);
   can->SetFrameBorderMode(0);
   can->SetFrameBorderMode(0);
   
   Double_t eff_graph_fx3001[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   17, 18, 19 };
   Double_t eff_graph_fy3001[20] = { 0.8315552, 0.8055365, 0.7645272, 0.7019058, 0.6179152, 0.5096822, 0.3985328, 0.2848657, 0.2015589, 0.1454082, 0.1009408, 0.0750471, 0.05671982, 0.04342224, 0.02970444, 0.02384842, 0.02090097,
   0.01685649, 0.01536313, 0.01331512 };
   Double_t eff_graph_felx3001[20] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
   0.5, 0.5, 0.5 };
   Double_t eff_graph_fely3001[20] = { 0.005360825, 0.003914465, 0.003697911, 0.003879097, 0.004235292, 0.004518722, 0.004521876, 0.004249899, 0.003841578, 0.003426485, 0.002982238, 0.002695468, 0.002468542, 0.002313436, 0.00206899, 0.001950032, 0.002037798,
   0.001942942, 0.002055589, 0.002117884 };
   Double_t eff_graph_fehx3001[20] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
   0.5, 0.5, 0.5 };
   Double_t eff_graph_fehy3001[20] = { 0.005360825, 0.003914465, 0.003697911, 0.003879097, 0.004235292, 0.004518722, 0.004521876, 0.004249899, 0.003841578, 0.003426485, 0.002982238, 0.002695468, 0.002468542, 0.002313436, 0.00206899, 0.001950032, 0.002037798,
   0.001942942, 0.002055589, 0.002117884 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(20,eff_graph_fx3001,eff_graph_fy3001,eff_graph_felx3001,eff_graph_fehx3001,eff_graph_fely3001,eff_graph_fehy3001);
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
   
   TH1F *Graph_eff_graph3001 = new TH1F("Graph_eff_graph3001","",100,-2.5,21.5);
   Graph_eff_graph3001->SetMinimum(0);
   Graph_eff_graph3001->SetMaximum(1.05);
   Graph_eff_graph3001->SetDirectory(nullptr);
   Graph_eff_graph3001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_eff_graph3001->SetLineColor(ci);
   Graph_eff_graph3001->GetXaxis()->SetTitle("Number of offline prompt tracks");
   Graph_eff_graph3001->GetXaxis()->SetRange(9,71);
   Graph_eff_graph3001->GetXaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetXaxis()->SetTitleSize(0.05);
   Graph_eff_graph3001->GetXaxis()->SetTitleOffset(1);
   Graph_eff_graph3001->GetXaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetYaxis()->SetTitle("HLT per jet tagging efficiency");
   Graph_eff_graph3001->GetYaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetYaxis()->SetTitleSize(0.05);
   Graph_eff_graph3001->GetYaxis()->SetTitleFont(42);
   Graph_eff_graph3001->GetZaxis()->SetLabelFont(42);
   Graph_eff_graph3001->GetZaxis()->SetTitleOffset(1);
   Graph_eff_graph3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_eff_graph3001);
   
   grae->Draw("ape");
   
   Double_t eff_graph_fx3002[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   17, 18, 19 };
   Double_t eff_graph_fy3002[20] = { 0.8074624, 0.789079, 0.7532616, 0.7032807, 0.6119705, 0.4771848, 0.3277849, 0.197732, 0.1258386, 0.07842292, 0.04922907, 0.03570199, 0.02785114, 0.01859034, 0.01348974, 0.00976529, 0.007614213,
   0.006954103, 0.005991285, 0.006472492 };
   Double_t eff_graph_felx3002[20] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
   0.5, 0.5, 0.5 };
   Double_t eff_graph_fely3002[20] = { 0.003871575, 0.003086464, 0.003178554, 0.003638924, 0.004296281, 0.004911036, 0.00484878, 0.004179108, 0.003450005, 0.00279025, 0.002270418, 0.001988007, 0.001802874, 0.001562087, 0.001396884, 0.001287113, 0.001214599,
   0.001265217, 0.001273514, 0.001442602 };
   Double_t eff_graph_fehx3002[20] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
   0.5, 0.5, 0.5 };
   Double_t eff_graph_fehy3002[20] = { 0.003871575, 0.003086464, 0.003178554, 0.003638924, 0.004296281, 0.004911036, 0.00484878, 0.004179108, 0.003450005, 0.00279025, 0.002270418, 0.001988007, 0.001802874, 0.001562087, 0.001396884, 0.001287113, 0.001214599,
   0.001265217, 0.001273514, 0.001442602 };
   grae = new TGraphAsymmErrors(20,eff_graph_fx3002,eff_graph_fy3002,eff_graph_felx3002,eff_graph_fehx3002,eff_graph_fely3002,eff_graph_fehy3002);
   grae->SetName("eff_graph");
   grae->SetTitle("Filter Efficiencey for Prompt Track Requirement");
   grae->SetFillColor(19);

   ci = TColor::GetColor("#0000ff");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#0000ff");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(21);
   grae->SetMarkerSize(1.5);
   
   TH1F *Graph_eff_graph3002 = new TH1F("Graph_eff_graph3002","Filter Efficiencey for Prompt Track Requirement",100,-2.5,21.5);
   Graph_eff_graph3002->SetMinimum(0.004245994);
   Graph_eff_graph3002->SetMaximum(0.8919956);
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
   TLatex *   tex = new TLatex(0.65,0.92,"(13.6 TeV)");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.15,0.82,"CMS #bf{#scale[0.75]{#it{Simulation}}}");
   tex->SetNDC();
   tex->SetTextSize(0.06);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.48,0.6,0.86,0.84,NULL,"brNDC");
   leg->SetBorderSize(1);
   leg->SetLineColor(0);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);

   ci = 1179;
   color = new TColor(ci, 1, 1, 1, " ", 0);
   leg->SetFillColor(ci);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("eff_graph","c#tau = 10 mm","ple");

   ci = TColor::GetColor("#00cc00");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#00cc00");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   entry=leg->AddEntry("eff_graph"," c#tau = 100 mm","ple");

   ci = TColor::GetColor("#0000ff");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#0000ff");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1.5);
   entry->SetTextFont(42);
   leg->Draw();
      tex = new TLatex(0.48,0.52,"#splitline{H #rightarrow SS, S #rightarrow b#bar{b}}{m_{H} = 125 GeV, m_{S} = 40 GeV}");
   tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.035);
   tex->SetLineWidth(2);
   tex->Draw();
   can->Modified();
   can->SetSelected(can);
}
