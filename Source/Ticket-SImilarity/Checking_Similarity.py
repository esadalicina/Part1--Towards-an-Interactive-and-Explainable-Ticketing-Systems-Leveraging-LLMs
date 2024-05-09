import pandas as pd
from tabulate import tabulate


# --------------------------- Create new dataset with examples of each category ----------------------------------------
# Initialize an empty DataFrame to store the new dataset
new_dataset = pd.DataFrame(columns=["Category", "Title", "Description"])


# ----------------------------------- Add solution examples to the new dataset ----------------------------------------

new_dataset['Category'] = ["Bank Account Services", "Bank Account Services", "Bank Account Services", "Bank Account Services", "Bank Account Services",
                           "Credit Card or Prepaid Card", "Credit Card or Prepaid Card", "Credit Card or Prepaid Card", "Credit Card or Prepaid Card", "Credit Card or Prepaid Card",
                            "Others", "Others", "Others", "Others", "Others",
                           "Theft/Dispute Reporting", "Theft/Dispute Reporting", "Theft/Dispute Reporting", "Theft/Dispute Reporting", "Theft/Dispute Reporting",
                           "Mortgage/Loan", "Mortgage/Loan", "Mortgage/Loan", "Mortgage/Loan", "Mortgage/Loan"]

new_dataset["Title"] = ["Online Banking", "Direct Deposit", "ATM Services", "Overdraft Protection", "Wire Transfers",
                        "Rewards Credit Card", "Secured Credit Card", "Prepaid Debit Card", "Travel Credit Card", "Student Credit Card",
                        "Investment Accounts", "Retirement Accounts", "Insurance Services", "Financial Planning Services", "Estate Planning Services",
                        "Unauthorized Transactions", "Identity Theft", "Fraudulent Charges", "Disputed Transactions", "Lost or Stolen Cards",
                        "Home Mortgage", "Auto Loan", "Personal Loan", "Student Loan", "Business Loan"]

new_dataset["Description"] = ["Online banking enables account holders to access their accounts, check balances, transfer funds, pay bills, and perform various other transactions through a bank's website or mobile application. It offers convenience and accessibility, allowing users to manage their finances from anywhere with internet access.",
                              "Direct deposit is a payment method where funds, such as paychecks, government benefits, or pension payments, are electronically deposited directly into a recipient's bank account. It eliminates the need for physical checks and provides faster access to funds, improving efficiency and security for both employers and recipients.",
                              "Automated Teller Machines (ATMs) provide convenient self-service options for banking transactions, including cash withdrawals, deposits, balance inquiries, and account transfers. ATMs are available at various locations, offering round-the-clock access to funds and basic banking services, even outside of regular banking hours.",
                              "Overdraft protection is a service offered by banks to prevent transactions that would overdraw an account, or to provide a line of credit to cover overdrafts. It helps account holders avoid costly overdraft fees and declined transactions, providing peace of mind and financial stability.",
                              "Wire transfers allow individuals and businesses to send or receive funds electronically between bank accounts domestically or internationally. They are commonly used for large transactions that require immediate settlement, such as real estate purchases, business payments, or international remittances.",
                              "Rewards credit cards offer incentives such as cashback, points, or airline miles for purchases made with the card. Cardholders can earn rewards based on their spending habits and redeem them for travel, merchandise, statement credits, or other perks.",
                              "Secured credit cards require a security deposit, which serves as collateral, and are designed for individuals with limited or damaged credit histories. They help users establish or rebuild credit by demonstrating responsible borrowing behavior and can eventually qualify for unsecured credit cards.",
                              "Prepaid debit cards allow users to load funds onto the card and use it for purchases, similar to a debit card, but without being linked to a bank account. They are convenient for budgeting, travel, and online shopping, offering spending control and security without the risk of overdraft fees.",
                              "Travel credit cards cater to frequent travelers and offer benefits such as travel insurance, airport lounge access, no foreign transaction fees, and bonus rewards for travel-related expenses. They provide valuable perks and rewards for individuals who frequently fly, stay in hotels, or dine out while traveling.",
                              "Student credit cards are tailored for college students with limited credit history and typically offer low credit limits, educational resources, and rewards or benefits geared towards students' needs and lifestyles. They help students build credit responsibly while managing their finances independently.",
                              "Investment accounts allow individuals to buy, sell, and hold investments such as stocks, bonds, mutual funds, and exchange-traded funds (ETFs). They provide opportunities for wealth accumulation, retirement savings, and portfolio diversification, helping investors achieve their financial goals.",
                              "Retirement accounts, such as 401(k)s, Individual Retirement Accounts (IRAs), and pensions, are designed to save and invest funds for retirement. They offer tax advantages, employer contributions (in the case of employer-sponsored plans), and various investment options to help individuals build a retirement nest egg.",
                              "Insurance services encompass various types of insurance coverage, including life insurance, health insurance, auto insurance, home insurance, and disability insurance. Insurance protects individuals and businesses against financial losses from unexpected events, such as accidents, illnesses, natural disasters, or death, providing financial security and peace of mind.",
                              "Financial planning services provide professional assistance with budgeting, investing, retirement planning, tax planning, estate planning, and other aspects of personal finance. Certified financial planners (CFPs) help individuals and families set financial goals, create customized financial plans, and make informed decisions to achieve financial success and security.",
                              "Estate planning services involve creating wills, trusts, powers of attorney, healthcare directives, and other legal documents to manage and distribute assets after death. Estate planning ensures that individuals' wishes are followed, minimizes estate taxes, avoids probate, and provides for the financial well-being of heirs and beneficiaries. It involves careful consideration of family dynamics, asset protection, and charitable giving strategies.",
                              "Reporting any unauthorized transactions on bank accounts or credit cards promptly to the financial institution to prevent further unauthorized access and initiate investigation and resolution processes.",
                              "Reporting instances of identity theft, where personal information has been stolen and used to open fraudulent accounts, obtain loans, or make unauthorized purchases, to law enforcement agencies, credit bureaus, and financial institutions to mitigate damages and restore affected individuals' credit and financial integrity.",
                              "Reporting suspicious or fraudulent charges on credit or debit card statements to the issuing bank or card issuer to dispute the charges, block the card if necessary, and investigate the unauthorized activity to prevent financial losses and identity theft.",
                              "Reporting discrepancies or errors in transactions, such as incorrect amounts charged, goods/services not received, or unauthorized charges, to the merchant or financial institution involved to rectify the errors, obtain refunds or credits, and protect consumers' rights and interests.",
                              "Reporting lost or stolen credit, debit, or prepaid cards immediately to the card issuer or financial institution to deactivate the card, prevent unauthorized use, and request a replacement card to minimize financial liability and prevent fraudulent transactions.",
                              "A home mortgage is a loan used to purchase a home or real estate property, with the property serving as collateral for the loan. It typically involves a down payment, principal amount, interest rate, and repayment period, such as 15, 20, or 30 years, and may offer fixed or adjustable interest rates.",
                              "An auto loan is a loan used to finance the purchase of a vehicle, such as a car, truck, or motorcycle, with the vehicle serving as collateral for the loan. It involves borrowing a specific amount of money from a lender, agreeing to repay it over a specified period, and paying interest on the outstanding balance until the loan is fully repaid.",
                              "A personal loan is an unsecured loan that can be used for various purposes, such as debt consolidation, home improvements, medical expenses, or major purchases. It does not require collateral and is typically based on the borrower's creditworthiness, income, and financial history, offering fixed or variable interest rates and flexible repayment terms.",
                              "A student loan is a loan specifically designed to finance education expenses, such as tuition, fees, books, and living expenses, for students attending college or university. It may be offered by the government (federal student loans) or private lenders and can be subsidized or unsubsidized, with varying interest rates, repayment options, and eligibility criteria.",
                              "A business loan is a loan used to start or expand a business, cover operating expenses, purchase equipment or inventory, or finance growth opportunities. It may be secured or unsecured, depending on the borrower's creditworthiness and the lender's risk assessment, and can be structured as term loans, lines of credit, equipment financing, or Small Business Administration (SBA) loans. Business loans typically require a detailed business plan, financial statements, and collateral (if applicable) to support the loan application and repayment."]


# Save the new dataset to a new CSV file
new_dataset.to_csv('../../Dataset/new_dataset.csv', index=False)

print(tabulate(new_dataset.head(), headers='keys', tablefmt='pretty'))



